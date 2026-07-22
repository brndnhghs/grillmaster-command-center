"""Blender Render — drive a live Blender instance over the Blender MCP socket.

Lets pipeline users drop a 3D-rendered frame into their node graph without
leaving the graph.  The node can render either a built-in primitive
(``source="primitive"``) or import a user model file
(``source="model_file"``, ``model_path=...`` — GLTF/GLB/OBJ/STL/FBX) and render
that.  Either way it builds a small Blender scene from its params (material,
light, camera, background, resolution, samples, engine), renders it headlessly
via the Blender MCP addon (port 9876), and returns the rendered PNG as a
canvas-sized float32 [0,1] (H,W,3) IMAGE.

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
from ..core.utils import save, mn, W, H, load_input


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


def _resolve_model_path(model_path: str) -> Path:
    """Resolve a user-supplied model path against the workspace or cwd.

    Accepts absolute paths as-is. Relative paths are tried against the
    project root and the current working directory so the user can pass a
    bare filename that lives next to the pipeline.
    """
    p = Path(model_path).expanduser()
    if p.is_absolute():
        return p
    here = Path.cwd()
    candidates = [here / model_path, Path(__file__).resolve().parents[2] / model_path]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the literal relative path (caller will surface a clear error).
    return p


def _build_texture_block(texture_mode: str, texture_path: str) -> str:
    """Return a Blender Python snippet (or '') wiring a wired IMAGE into 3D.

    Honors the pipeline's Wired-Input Override pattern (skill Rule 12): when an
    upstream 2D IMAGE is wired into this node it is brought *into* 3D.

    texture_mode:
      * ``'none'`` -> '' (no wiring)
      * ``'decal'`` -> map the image onto the object's surface (Image Texture
        node driven by the object's UVs).  Guarded so it only applies when a
        ``mat``/``bsdf`` exists (i.e. the primitive path, or a model import with
        ``apply_material=True``).
      * ``'env'``   -> use the image as the World environment backdrop
        (Environment Texture node).  Independent of geometry, so it works for
        every source mode.
    """
    if not texture_mode or texture_mode == "none" or not texture_path:
        return ""
    tex = str(texture_path).replace("\\", "/")
    if texture_mode == "decal":
        return f'''
# ── Wired IMAGE as object decal ──
if 'mat' in locals() and mat is not None and hasattr(mat, 'node_tree'):
    _img = bpy.data.images.load(r"{tex}")
    _tn = mat.node_tree.nodes.new("ShaderNodeTexImage")
    _tn.image = _img
    _tn.location = (-380, 220)
    _uv = mat.node_tree.nodes.new("ShaderNodeUVMap")
    _uv.location = (-640, 220)
    mat.node_tree.links.new(_uv.outputs[0], _tn.inputs[0])
    try:
        mat.node_tree.links.new(_tn.outputs[0], bsdf.inputs["Base Color"])
        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass
'''
    # 'env' -> 3D environment backdrop (non-destructive: link the env texture to
    # the world Output's Surface.  The world tree was reset earlier in the
    # script to a clean Background+Output, so `_wn` and `_out` exist and the
    # Background node is preserved for the next render.)
    return f'''
# ── Wired IMAGE as 3D environment backdrop ──
_img = bpy.data.images.load(r"{tex}")
_env = _wn.nodes.new("ShaderNodeTexEnvironment")
_env.image = _img
_env.location = (-460, 0)
_wn.links.new(_env.outputs[0], _out.inputs["Surface"])
'''


def _build_model_script(  # noqa: C901
    model_path: str,
    apply_material: bool,
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
    texture_mode: str = "none",
    texture_path: str = "",
) -> str:
    """Return a Blender Python script that imports model_path and renders it."""
    r, g, b = _hex_to_rgb01(color_hex)
    br, bg, bb = _hex_to_rgb01(bg_hex, "#0a0e18")
    eng = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE"
    samples = max(1, int(samples))
    spin = float(spin_deg)
    mp = str(model_path).replace("\\", "/")
    cam_z = max(2.0, size * 3.2)
    cam_y = max(1.5, size * 1.2)
    texture_block = _build_texture_block(texture_mode, texture_path)

    # Import strategy:
    #  * OBJ / STL are parsed directly in Python and turned into a mesh via
    #    bpy.data.meshes.new().from_pydata() — NO operator, so it never hits
    #    the operator poll()/context requirements (and needs no addon).
    #    This is robust even when the corresponding Blender addon is not
    #    installed (e.g. io_scene_obj is absent on some Blender builds).
    #  * GLTF / GLB / FBX use the stock operators, wrapped in a full
    #    VIEW_3D context override so their poll() passes.
    import_block = (
        "def _ctx_override():\n"
        "    win = bpy.data.window_managers[0].windows[0]\n"
        "    sc = win.screen\n"
        "    area = next((a for a in sc.areas if a.type == 'VIEW_3D'), None)\n"
        "    region = next((r for r in area.regions if r.type == 'WINDOW'), None)\n"
        "    space = next((sp for sp in area.spaces if sp.type == 'VIEW_3D'), None)\n"
        "    return {'area': area, 'region': region, 'space_data': space,\n"
        "            'screen': sc, 'scene': bpy.context.scene,\n"
        "            'view_layer': bpy.context.view_layer, 'window': win}\n"
        "\n"
        "def _parse_obj(path):\n"
        "    verts = []\n"
        "    faces = []\n"
        "    with open(path) as fh:\n"
        "        for line in fh:\n"
        "            if line.startswith('v '):\n"
        "                p = line[2:].split()\n"
        "                verts.append((float(p[0]), float(p[1]), float(p[2])))\n"
        "            elif line.startswith('f '):\n"
        "                idx = [int(t.split('/')[0]) for t in line[2:].split() if t.split('/')[0]]\n"
        "                if len(idx) >= 3:\n"
        "                    faces.append(tuple(i - 1 for i in idx))\n"
        "    if not verts or not faces:\n"
        "        raise RuntimeError('Blender Render: no geometry parsed from OBJ ' + path)\n"
        "    return verts, faces\n"
        "\n"
        "def _parse_stl(path):\n"
        "    import struct\n"
        "    verts = []\n"
        "    faces = []\n"
        "    with open(path, 'rb') as fh:\n"
        "        header = fh.read(80)\n"
        "        if header[:5].lower() == b'solid':\n"
        "            raise RuntimeError('Blender Render: ASCII STL is not supported; use binary STL')\n"
        "        (n,) = struct.unpack('<I', fh.read(4))\n"
        "        data = fh.read(50 * n)\n"
        "    off = 0\n"
        "    for _ in range(n):\n"
        "        fh2 = data[off + 12:off + 48]\n"
        "        tri = struct.unpack('<9f', fh2)\n"
        "        base = len(verts)\n"
        "        verts.append((tri[0], tri[1], tri[2]))\n"
        "        verts.append((tri[3], tri[4], tri[5]))\n"
        "        verts.append((tri[6], tri[7], tri[8]))\n"
        "        faces.append((base, base + 1, base + 2))\n"
        "        off += 50\n"
        "    if not verts:\n"
        "        raise RuntimeError('Blender Render: no geometry parsed from STL ' + path)\n"
        "    return verts, faces\n"
        "\n"
        "def _build_mesh(verts, faces, name):\n"
        "    me = bpy.data.meshes.new(name)\n"
        "    me.from_pydata(verts, [], list(faces))\n"
        "    me.validate()\n"
        "    me.update()\n"
        "    obj = bpy.data.objects.new(name, me)\n"
        "    bpy.context.scene.collection.objects.link(obj)\n"
        "    return obj\n"
        "\n"
        "def _import_model(mp):\n"
        "    low = mp.lower()\n"
        "    if low.endswith('.obj'):\n"
        "        v, f = _parse_obj(mp)\n"
        "        _build_mesh(v, f, 'ImportedOBJ')\n"
        "    elif low.endswith('.stl'):\n"
        "        v, f = _parse_stl(mp)\n"
        "        _build_mesh(v, f, 'ImportedSTL')\n"
        "    else:\n"
        "        ctx = _ctx_override()\n"
        "        with bpy.context.temp_override(**ctx):\n"
        "            if low.endswith('.gltf') or low.endswith('.glb'):\n"
        "                bpy.ops.import_scene.gltf(filepath=mp)\n"
        "            elif low.endswith('.fbx'):\n"
        "                bpy.ops.import_scene.fbx(filepath=mp)\n"
        "            else:\n"
        "                raise RuntimeError('Blender Render: unsupported model extension: ' + mp)\n"
    )

    material_block = ""
    if apply_material:
        material_block = (
            "mat = bpy.data.materials.new(name=\"GrillMat\")\n"
            "mat.use_nodes = True\n"
            "bsdf = mat.node_tree.nodes.get(\"Principled BSDF\")\n"
            "if bsdf is not None:\n"
            "    bsdf.inputs[\"Base Color\"].default_value = ({r}, {g}, {b}, 1.0)\n"
            "    bsdf.inputs[\"Metallic\"].default_value = {metalness}\n"
            "    bsdf.inputs[\"Roughness\"].default_value = {roughness}\n"
            "for o in imported:\n"
            "    if o.type == \"MESH\":\n"
            "        o.data.materials.clear()\n"
            "        o.data.materials.append(mat)\n"
        ).format(r=r, g=g, b=b, metalness=metalness, roughness=roughness)

    return f"""
import bpy, os, math

# ── Wipe scene ──
for o in list(bpy.data.objects):
    if o.type in ("MESH", "LIGHT", "CAMERA", "EMPTY"):
        bpy.data.objects.remove(o, do_unlink=True)
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m, do_unlink=True)

# ── Reset World node tree to a known state ──
_wn = bpy.context.scene.world.node_tree
for _n in list(_wn.nodes):
    _wn.nodes.remove(_n)
_bg = _wn.nodes.new("ShaderNodeBackground")
_bg.location = (-200, 0)
_out = _wn.nodes.new("ShaderNodeOutputWorld")
_out.location = (200, 0)
_wn.links.new(_bg.outputs[0], _out.inputs["Surface"])

# ── Import model ──
mp = r"{mp}"
{import_block}
_import_model(mp)
imported = [o for o in bpy.data.objects if o.type == "MESH"]
if not imported:
    raise RuntimeError("Blender Render: no mesh objects imported from " + mp)

# Normalize: center on origin, scale to fit, then apply user scale.
def _bbox(objs):
    mins = [1e9, 1e9, 1e9]
    maxs = [-1e9, -1e9, -1e9]
    for o in objs:
        for v in o.bound_box:
            for i in range(3):
                mins[i] = min(mins[i], o.matrix_world[i][3] + v[i])
                maxs[i] = max(maxs[i], o.matrix_world[i][3] + v[i])
    return mins, maxs
mins, maxs = _bbox(imported)
center = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
extent = max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]) or 1.0
fit = 2.0 / extent
for o in imported:
    o.location[0] -= center[0]
    o.location[1] -= center[1]
    o.location[2] -= center[2]
    o.scale = (o.scale[0] * fit * {size},
               o.scale[1] * fit * {size},
               o.scale[2] * fit * {size})
    # Spin about X (not Y): the camera views along +Y, so a Y-spin would be a
    # turntable seen head-on and barely move.  X tumbles the model visibly.
    o.rotation_euler = ({spin} * math.pi / 180.0, 0.0, 0.0)

# Apply the node's PBR material to every imported mesh when requested.
{material_block}

# NOTE: we rotate the imported mesh about Y (see the normalize loop above) so
# the animation is visible.  A Z-camera-orbit would be invisible for
# Z-symmetric models, so mesh Y-rotation is the correct strategy.

# ── Light ──
bpy.ops.object.light_add(type="POINT", location=(4, -4, 6))
lit = next((o for o in bpy.data.objects if o.type == "LIGHT"), None)
if lit is not None:
    lit.data.energy = {light_intensity}
    lit.data.use_shadow = True

# ── Camera (fixed 3/4 view; the MESH spins, not the camera) ──
# We rotate the mesh about Y (see the mesh block).  Orbiting the camera about
# the Z axis produces zero visible change for Z-symmetric primitives (torus,
# cylinder, cone, sphere) and is therefore the wrong animation strategy.
import mathutils
bpy.ops.object.camera_add(location=(0.0, -{cam_z}, {cam_y}))
cam = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
if cam is not None:
    cam.data.lens = 35
    _dir = mathutils.Vector((0.0, 0.0, 0.0)) - cam.location
    cam.rotation_euler = _dir.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

# ── World / background ──
bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = ({br}, {bg}, {bb}, 1.0)

# ── Wired IMAGE into 3D (decal on this mesh / env backdrop) ──
# Runs after the material + world are set, so the 'env' branch may safely wipe
# and rebuild the World node tree without breaking the Background lookup above.
{texture_block}

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
    texture_mode: str = "none",
    texture_path: str = "",
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
    texture_block = _build_texture_block(texture_mode, texture_path)

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

# ── Reset World node tree to a known state ──
# Blender's world tree is *persistent* across renders in the same session, so a
# previous (e.g. 'env') render could leave it mutated.  Rebuild it deterministically
# to a clean Background + OutputWorld every time, and surface handles for the
# wired-image 'env' backdrop branch.
_wn = bpy.context.scene.world.node_tree
for _n in list(_wn.nodes):
    _wn.nodes.remove(_n)
_bg = _wn.nodes.new("ShaderNodeBackground")
_bg.location = (-200, 0)
_out = _wn.nodes.new("ShaderNodeOutputWorld")
_out.location = (200, 0)
_wn.links.new(_bg.outputs[0], _out.inputs["Surface"])

# ── Mesh ──
{ctor.format(size=size)}
obj = next((o for o in bpy.data.objects if o.type == "MESH"), None)
if obj is None:
    raise RuntimeError("Blender Render: mesh primitive failed to create object")
# NOTE: the mesh is spun about X (see the mesh-spin block below).  The camera
# views along +Y, so a Y-spin would be a turntable seen head-on (near-invisible).
# An X-spin tumbles the object and is clearly visible for every non-spherical
# primitive; a uniform sphere is spin-static by symmetry (expected).

# ── Material (Principled BSDF) ──
mat = bpy.data.materials.new(name="GrillMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf is not None:
    bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, 1.0)
    bsdf.inputs["Metallic"].default_value = {metalness}
    bsdf.inputs["Roughness"].default_value = {roughness}
obj.data.materials.append(mat)

# ── Mesh spin (about X) ──
# The camera is placed at (0, -cam_z, cam_y) looking toward the origin, i.e. its
# view axis is essentially +Y.  Spinning the mesh about Y would therefore rotate
# it around the *camera's view axis* (a turntable seen head-on) and show almost
# no motion.  Spinning about X (the screen-horizontal axis) tumbles the object
# and is clearly visible for every primitive including torus/cube/monkey.
# NOTE: a uniform sphere is spherically symmetric, so any spin is inherently
# static — that is expected, not a bug.
obj.rotation_euler = ({spin} * math.pi / 180.0, 0.0, 0.0)

# ── Light ──
bpy.ops.object.light_add(type="POINT", location=(4, -4, 6))
lit = next((o for o in bpy.data.objects if o.type == "LIGHT"), None)
if lit is not None:
    lit.data.energy = {light_intensity}
    lit.data.use_shadow = True

# ── Camera (fixed 3/4 view; the MESH spins, not the camera) ──
# We rotate the mesh about Y (see the mesh block).  Orbiting the camera about
# the Z axis produces zero visible change for Z-symmetric primitives (torus,
# cylinder, cone, sphere) and is therefore the wrong animation strategy.
import mathutils
bpy.ops.object.camera_add(location=(0.0, -{cam_z}, {cam_y}))
cam = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
if cam is not None:
    cam.data.lens = 35
    _dir = mathutils.Vector((0.0, 0.0, 0.0)) - cam.location
    cam.rotation_euler = _dir.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

# ── World / background ──
bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = ({br}, {bg}, {bb}, 1.0)

# ── Wired IMAGE into 3D (decal on this mesh / env backdrop) ──
# Runs after the material + world are set, so the 'env' branch may safely wipe
# and rebuild the World node tree without breaking the Background lookup above.
{texture_block}

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
    inputs={"image_in": "IMAGE"},  # optional wire — a 2D IMAGE maps into 3D
    outputs={"image": "IMAGE", "field": "FIELD"},
    params={
        "source": {
            "description": (
                "what to render — a built-in primitive, or import a model "
                "file (GLTF/GLB/OBJ/STL/FBX) from model_path"
            ),
            "default": "primitive",
            "choices": ["primitive", "model_file"],
        },
        "shape": {
            "description": "primitive geometry to render (used when source=primitive)",
            "default": "torus",
            "choices": [
                "torus", "sphere", "ico_sphere", "cube", "cylinder",
                "cone", "monkey", "plane",
            ],
        },
        "model_path": {"content": True, 
            "description": (
                "absolute or workspace-relative path to a model file "
                "(GLTF/GLB/OBJ/STL/FBX). Ignored unless source=model_file."
            ),
            "default": "",
        },
        "apply_material": {
            "description": (
                "when true (and source=model_file) the node's PBR material "
                "is applied to the imported mesh; when false the model's own "
                "materials are kept"
            ),
            "default": False,
            "choices": [True, False],
        },
        "size": {
            "description": "overall scale of the mesh (primitive) or uniform scale of the imported model",
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
        "texture_mode": {
            "description": (
                "how a wired upstream IMAGE is brought into 3D: "
                "'decal' maps it onto the object surface (UVs), "
                "'env' uses it as a 3D environment backdrop, "
                "'none' ignores the wire"
            ),
            "default": "none",
            "choices": ["none", "decal", "env"],
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

    source = str(params.get("source", "primitive"))
    shape = str(params.get("shape", "torus"))
    model_path = str(params.get("model_path", ""))
    apply_material = bool(params.get("apply_material", False))
    size = float(params.get("size", 1.0))
    color = str(params.get("color", "#4a9eff"))
    metalness = float(params.get("metalness", 0.4))
    roughness = float(params.get("roughness", 0.35))
    bg_color = str(params.get("bg_color", "#0a0e18"))
    light_intensity = float(params.get("light_intensity", 120.0))
    engine = str(params.get("engine", "cycles"))
    samples = int(params.get("samples", 64))
    spin_speed = float(params.get("spin_speed", 0.0))
    texture_mode = str(params.get("texture_mode", "none"))

    cw, ch = int(W), int(H)

    # ── Wired IMAGE override (skill Rule 12) ──
    # If an upstream IMAGE is wired in, save it to a PNG and bring it into 3D
    # via texture_mode. We ALWAYS honor the wire over the solid-color material
    # path. When no image is wired, texture_mode stays 'none' and the node
    # renders its normal procedural scene.
    wired_input_path = str(params.get("input_image", ""))
    texture_path = ""
    if wired_input_path:
        try:
            img_arr = load_input(wired_input_path, cw, ch)
            tmp_tx = Path(tempfile.mkdtemp(prefix="blender_tx_")) / "_wired_input.png"
            Image.fromarray((img_arr * 255).astype(np.uint8), "RGB").save(str(tmp_tx))
            texture_path = str(tmp_tx)
            if texture_mode == "none":
                texture_mode = "decal"  # sensible default when wired
        except (FileNotFoundError, OSError, ValueError):
            texture_path = ""  # fall through to normal procedural render

    # Injected timeline frame (executor sets params["frame"] per frame).
    frame = int(params.get("frame", 0))
    spin_deg = spin_speed * frame

    if source == "model_file":
        if not model_path:
            raise ValueError(
                "Blender Render: source=model_file but model_path is empty."
            )
        resolved = _resolve_model_path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Blender Render: model file not found: {resolved}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            out_png = Path(tmp) / "_blender_render.png"
            script = _build_model_script(
                model_path=str(resolved),
                apply_material=apply_material,
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
                texture_mode=texture_mode,
                texture_path=texture_path,
            )
            reply = _blender_exec(script, timeout=180.0)
            status = reply.get("status")
            if status != "success":
                msg = reply.get("message") or str(reply)[:300]
                raise RuntimeError(f"Blender Render: Blender MCP error: {msg}")
            if not out_png.exists():
                raise RuntimeError(
                    "Blender Render: Blender finished but no PNG was written "
                    f"(expected {out_png}). Check the Blender console for "
                    "import/render errors."
                )
            img = Image.open(str(out_png)).convert("RGB")
            if img.size != (cw, ch):
                img = img.resize((cw, ch), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
        save(arr, mn(0, "Blender Render"), out_dir)
        return {"image": arr, "field": arr}

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
            texture_mode=texture_mode,
            texture_path=texture_path,
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
