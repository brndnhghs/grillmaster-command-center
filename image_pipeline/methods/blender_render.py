"""Blender Render — drive a live Blender instance over the Blender MCP socket.

Lets pipeline users drop a 3D-rendered frame into their node graph without
leaving the graph.  The node builds a small Blender scene from its params
(shape, material, light, camera, background, resolution, samples, engine),
renders it headlessly via the Blender MCP addon (port 9876), and returns
the rendered PNG as a canvas-sized float32 [0,1] (H,W,3) IMAGE.

Prerequisites (all checked at runtime, no hard import):
  * Blender desktop app running with the "Blender MCP" addon enabled,
    and its socket server started (N-panel > BlenderMCP > Start Server).

This is a SOURCE node: it owns ``inputs={}`` (no upstream image port)
and emits a single IMAGE + FIELD, slotting in anywhere a generator would.

Architecture: single-shot per cook (``is_time_varying=False`` by default —
the output only changes when a param changes).  When wired into an animated
graph, the same params yield the same image every frame.  Set ``spin_speed``
> 0 to make the render advance per frame (the node reads the injected
``frame`` param and rotates the mesh), in which case the executor re-cooks
it each frame; we declare ``is_time_varying=True`` so that happens.

Safety: only additive — does NOT touch server routing or the 3D sidecar.
"""
from __future__ import annotations

import socket
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.registry import method
from ..core.utils import save, mn, W, H


# ── Blender MCP socket client (self-contained, no new dependency) ──────────

_BLENDER_HOST = "localhost"
_BLENDER_PORT = 9876


def _blender_exec(code: str, timeout: float = 60.0) -> dict:
    """Send a `execute_code` command to the Blender MCP socket and return the JSON reply.

    Raises ConnectionError if Blender MCP is not reachable so the node surfaces a
    clear, actionable error instead of silently producing a black frame.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((_BLENDER_HOST, _BLENDER_PORT))
    except (ConnectionRefusedError, OSError) as e:
        raise ConnectionError(
            "Blender Render: cannot reach Blender MCP at "
            f"{_BLENDER_HOST}:{_BLENDER_PORT}. Open Blender, enable the "
            "'Blender MCP' addon, and click Start Server in the sidebar."
        ) from e

    payload = json.dumps({"type": "execute_code", "params": {"code": code}})
    s.sendall(payload.encode("utf-8"))

    buf = b""
    while True:
        try:
            chunk = s.recv(8192)
        except socket.timeout as e:
            raise TimeoutError(
                "Blender Render: Blender MCP did not respond within "
                f"{timeout:.0f}s (scene may be too heavy for the sample count)."
            ) from e
        if not chunk:
            break
        buf += chunk
        try:
            return json.loads(buf.decode("utf-8"))
        except json.JSONDecodeError:
            # Wait for the rest of a split frame.
            continue
    s.close()
    try:
        return json.loads(buf.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Blender Render: malformed reply from Blender MCP: {buf[:200]!r}"
        ) from e


def _read_png_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ── Param → scene builder ──────────────────────────────────────────────────

_SHAPE_CTOR = {
    "sphere": "bpy.ops.mesh.primitive_uv_sphere_add(radius={size}, location=(0,0,0))",
    "cube": "bpy.ops.mesh.primitive_cube_add(size={size}, location=(0,0,0))",
    "cylinder": "bpy.ops.mesh.primitive_cylinder_add(radius={size}*0.6, depth={size}*1.6, location=(0,0,0))",
    "cone": "bpy.ops.mesh.primitive_cone_add(radius1={size}*0.7, radius2=0.0, depth={size}*1.6, location=(0,0,0))",
    "torus": "bpy.ops.mesh.primitive_torus_add(major_radius={size}*0.7, minor_radius={size}*0.28, location=(0,0,0))",
    "ico_sphere": "bpy.ops.mesh.primitive_ico_sphere_add(radius={size}, subdivisions=2, location=(0,0,0))",
    "monkey": "bpy.ops.mesh.primitive_monkey_add(size={size}, location=(0,0,0))",
    "plane": "bpy.ops.mesh.primitive_plane_add(size={size}*2.0, location=(0,0,0))",
}


def _hex_to_rgb01(hex_str: str, default: str = "#4a9eff") -> tuple[float, float, float]:
    h = (hex_str or default).lstrip("#").strip()
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = default.lstrip("#")
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
    except ValueError:
        r, g, b = (0.29, 0.62, 1.0)
    return (r, g, b)


def _build_scene_script(
    shape: str,
    size: float,
    color_hex: str,
    metalness: float,
    roughness: float,
    bg_hex: str,
    light_intensity: float,
    samples: int,
    engine: str,
    res_x: int,
    res_y: int,
    spin_deg: float,
    out_path: str,
) -> str:
    """Return a Blender Python script (as a string) that renders to out_path."""
    ctor = _SHAPE_CTOR.get(shape, _SHAPE_CTOR["torus"])
    r, g, b = _hex_to_rgb01(color_hex)
    br, bg, bb = _hex_to_rgb01(bg_hex, "#0a0e18")
    eng = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE"
    samples = max(1, int(samples))
    spin = float(spin_deg)
    cam_z = max(2.0, size * 3.2)
    cam_y = max(1.5, size * 1.2)

    # The Blender MCP exec context does not expose bpy.context.active_object,
    # so we create then fetch the single object of each type by iterating
    # bpy.data.objects (the scene is wiped first, so each add yields exactly
    # one new mesh / light / camera).
    return f"""
import bpy, os, math

# ── Wipe scene ──
for o in list(bpy.data.objects):
    if o.type in ("MESH", "LIGHT", "CAMERA", "EMPTY"):
        bpy.data.objects.remove(o, do_unlink=True)
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m, do_unlink=True)

# ── Mesh ──
{ctor.format(size=size)}
obj = next((o for o in bpy.data.objects if o.type == "MESH"), None)
if obj is None:
    raise RuntimeError("Blender Render: mesh primitive failed to create object")
# Spin about Y — a non-symmetry axis for the default torus/cube/monkey
# (Z is the torus symmetry axis, so a Z-spin would be invisible).
obj.rotation_euler = (0.0, {spin} * math.pi / 180.0, 0.0)

# ── Material (Principled BSDF) ──
mat = bpy.data.materials.new(name="GrillMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf is not None:
    bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, 1.0)
    bsdf.inputs["Metallic"].default_value = {metalness}
    bsdf.inputs["Roughness"].default_value = {roughness}
obj.data.materials.append(mat)

# ── Light ──
bpy.ops.object.light_add(type="POINT", location=(4, -4, 6))
lit = next((o for o in bpy.data.objects if o.type == "LIGHT"), None)
if lit is not None:
    lit.data.energy = {light_intensity}
    lit.data.use_shadow = True

# ── Camera ──
bpy.ops.object.camera_add(location=(0, -{cam_z}, {cam_y}))
cam = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
if cam is not None:
    cam.data.lens = 35
    cam.rotation_euler = (math.radians(63.0), 0.0, 0.0)
    bpy.context.scene.camera = cam

# ── World / background ──
bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = ({br}, {bg}, {bb}, 1.0)

# ── Render settings ──
scn = bpy.context.scene
scn.render.engine = "{eng}"
scn.render.resolution_x = {res_x}
scn.render.resolution_y = {res_y}
scn.render.resolution_percentage = 100
scn.render.filepath = r"{out_path}"
if "{engine}" == "cycles":
    scn.cycles.samples = {samples}
    scn.cycles.device = "CPU"
scn.render.film_transparent = False
bpy.ops.render.render(write_still=True)

print("BLENDER_RENDER_DONE path=%s exists=%s" % (r"{out_path}", os.path.exists(r"{out_path}")))
"""


# ── The method ──────────────────────────────────────────────────────────────

@method(
    id="__blender_render__",
    name="Blender Render",
    category="client_3d",
    tags=["3d", "blender", "render", "source", "external"],
    new_image_contract=True,
    # Spin > 0 makes output depend on the frame → re-cook each frame.
    is_time_varying=True,
    inputs={},  # source node — no image_in port
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "shape": {
            "description": "primitive geometry to render",
            "default": "torus",
            "choices": [
                "torus", "sphere", "ico_sphere", "cube", "cylinder",
                "cone", "monkey", "plane",
            ],
        },
        "size": {
            "description": "overall scale of the mesh",
            "min": 0.3, "max": 3.0, "default": 1.0,
        },
        "color": {
            "description": "material base color (hex)",
            "default": "#4a9eff",
        },
        "metalness": {
            "description": "PBR metalness 0=plastic 1=metal",
            "min": 0.0, "max": 1.0, "default": 0.4,
        },
        "roughness": {
            "description": "PBR surface roughness",
            "min": 0.0, "max": 1.0, "default": 0.35,
        },
        "bg_color": {
            "description": "scene background color (hex)",
            "default": "#0a0e18",
        },
        "light_intensity": {
            "description": "point light energy",
            "min": 5.0, "max": 500.0, "default": 120.0,
        },
        "engine": {
            "description": "render engine",
            "default": "cycles",
            "choices": ["cycles", "eevee"],
        },
        "samples": {
            "description": "Cycles samples (higher=cleaner, slower)",
            "min": 8, "max": 512, "default": 64,
        },
        "spin_speed": {
            "description": "mesh Y-rotation per frame in degrees (0 = static)",
            "min": 0.0, "max": 60.0, "default": 0.0,
        },
    },
)
def method_blender_render(out_dir: Path, seed: int, params=None):
    """Render a 3D scene in live Blender via MCP and emit it as an IMAGE.

    Outputs:
        image (IMAGE): the Blender render, canvas-sized
        field (FIELD): the same array, for FIELD-input nodes
    """
    if params is None:
        params = {}

    shape = str(params.get("shape", "torus"))
    size = float(params.get("size", 1.0))
    color = str(params.get("color", "#4a9eff"))
    metalness = float(params.get("metalness", 0.4))
    roughness = float(params.get("roughness", 0.35))
    bg_color = str(params.get("bg_color", "#0a0e18"))
    light_intensity = float(params.get("light_intensity", 120.0))
    engine = str(params.get("engine", "cycles"))
    samples = int(params.get("samples", 64))
    spin_speed = float(params.get("spin_speed", 0.0))

    # Injected timeline frame (executor sets params["frame"] per frame).
    frame = int(params.get("frame", 0))
    spin_deg = spin_speed * frame

    cw, ch = int(W), int(H)

    with tempfile.TemporaryDirectory() as tmp:
        out_png = Path(tmp) / "_blender_render.png"
        script = _build_scene_script(
            shape=shape,
            size=size,
            color_hex=color,
            metalness=metalness,
            roughness=roughness,
            bg_hex=bg_color,
            light_intensity=light_intensity,
            samples=samples,
            engine=engine,
            res_x=cw,
            res_y=ch,
            spin_deg=spin_deg,
            out_path=str(out_png),
        )
        reply = _blender_exec(script, timeout=120.0)
        status = reply.get("status")
        if status != "success":
            msg = reply.get("message") or str(reply)[:300]
            raise RuntimeError(f"Blender Render: Blender MCP error: {msg}")

        if not out_png.exists():
            raise RuntimeError(
                "Blender Render: Blender finished but no PNG was written "
                f"(expected {out_png}). Check the Blender console for render errors."
            )

        # Read back as float32 [0,1] and resize to canvas (already at canvas
        # resolution, but resize is a cheap safety net for rounding).
        img = Image.open(str(out_png)).convert("RGB")
        if img.size != (cw, ch):
            img = img.resize((cw, ch), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

    save(arr, mn(0, "Blender Render"), out_dir)
    return {"image": arr, "field": arr}
