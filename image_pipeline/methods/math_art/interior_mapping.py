"""
#525 — Interior Mapping (fake 3D rooms behind a flat facade)

A numpy re-implementation of Joost van Dongen's *Interior Mapping* real-time
shader technique (Computer Graphics International 2008,
https://www.proun-game.com/Oogst3D/CODING/InteriorMapping/InteriorMapping.pdf).

The idea: render the *interior* of a building — rooms with a back wall, floor,
ceiling and side walls — while the actual geometry is just a flat plane (the
facade). For every facade pixel a view ray is cast into the wall and intersected
against a small set of virtual interior planes (the room box). The nearest
positive hit tells us which interior surface the eye sees *through* that pixel,
and we shade it. No interior geometry is stored — the rooms are pure ray-plane
math, so a whole skyscraper of distinct rooms costs one fragment shader.

This node lays a regular grid of windows across the frame (a facade), gives each
window its own hashed room (depth, wall tint, and a lit/unlit state), and casts
a per-pixel ray into the room box:

    origin  o = (lx, ly, 0)          # on the window plane, local window coords
    dir     d = (lx*persp + pan_x,   # parallax grows with screen offset
                 ly*persp + pan_y,
                 1)                    # +z points into the room
    box     x,y in [-0.5, 0.5],  back wall at z = room_depth

The origin is on the front face of the box, so the ray exits through exactly one
of the far planes (back wall / a side wall / floor / ceiling) — we take the
nearest positive t. Depth attenuation darkens deeper surfaces; the back wall
carries a procedural "furniture / window" pattern and a ceiling light glow sells
the room. Because the view direction depends on each window's position on the
facade, the parallax shifts window-to-window exactly like the real technique.

Animation modes (closed-form f(uv, t), cheap O(W·H) numpy → never hits the 150 s
render-timeout cull; Architecture A internal frame loop):

    none:    static facade (time ignored → Δ≈0)
    pan:     the virtual camera pans across the building (parallax sweep)
    lights:  windows flicker their lit/unlit state (evening building)
    zoom:    the camera dollies, changing perspective strength (fov breathe)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H,
    write_scalars, write_field, write_mask, wired_source_lum,
)
from ...core.animation import capture_frame

TAU = 2.0 * math.pi


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Vectorised integer hash → float in [0,1). Stable per (cell, seed)."""
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    h = (ix * 374761393 + iy * 668265263 + np.int64(seed) * 2654435761) & 0x7FFFFFFF
    h = (h ^ (h >> 13))
    h = (h * 1274126177) & 0x7FFFFFFF
    h = (h ^ (h >> 16))
    return (h & 0x00FFFFFF).astype(np.float64) / 16777216.0


def _compute_interior(
    w: int, h: int, t: float, p: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one interior-mapped facade frame.

    Returns (rgb HxWx3 float32 [0,1], depth-normalised field HxW float32).
    `t` is the animation phase in radians (t=0 for anim_mode "none").
    """
    anim_mode = str(p.get("anim_mode", "none"))
    anim_speed = float(p.get("anim_speed", 1.0))
    n_cols = max(1, int(round(float(p.get("n_cols", 8.0)))))
    n_rows = max(1, int(round(float(p.get("n_rows", 6.0)))))
    room_depth = float(p.get("room_depth", 1.4))
    persp = float(p.get("perspective", 1.1))
    pan_x0 = float(p.get("pan_x", 0.0))
    pan_y0 = float(p.get("pan_y", 0.15))
    frame_w = float(p.get("frame_width", 0.06))
    lit_frac = float(p.get("lit_fraction", 0.6))
    palette_warm = float(p.get("warmth", 0.5))
    seed = int(p.get("seed", 0))

    tt = 0.0 if anim_mode == "none" else t * anim_speed

    # ── Animate the virtual camera / lights ──
    pan_x, pan_y = pan_x0, pan_y0
    persp_eff = persp
    lit_phase = 0.0
    if anim_mode == "pan":
        pan_x = pan_x0 + 0.6 * math.sin(tt)
        pan_y = pan_y0 + 0.25 * math.cos(tt * 0.7)
    elif anim_mode == "zoom":
        persp_eff = persp * (0.6 + 0.5 * (0.5 + 0.5 * math.sin(tt)))
    elif anim_mode == "lights":
        lit_phase = tt

    # ── Pixel grid → per-window local coords ──
    ys, xs = np.mgrid[0:h, 0:w]
    # Global uv, aspect-corrected so windows aren't stretched.
    gu = (xs + 0.5) / w
    gv = (ys + 0.5) / h

    fu = gu * n_cols
    fv = gv * n_rows
    ci = np.floor(fu).astype(np.int64)   # window column index
    cj = np.floor(fv).astype(np.int64)   # window row index
    lx = (fu - ci) - 0.5                 # local window coord in [-0.5, 0.5]
    ly = (fv - cj) - 0.5

    # ── Facade mullions / window frame (drawn last, but detect region now) ──
    in_frame = (np.abs(lx) > (0.5 - frame_w)) | (np.abs(ly) > (0.5 - frame_w))

    # ── Per-window hashed room parameters ──
    rd = _hash2(ci, cj, seed + 1)                    # depth jitter
    depth = room_depth * (0.75 + 0.5 * rd)
    wall_h = _hash2(ci, cj, seed + 2)                # wall hue selector
    lit_h = _hash2(ci, cj, seed + 3)                 # lit/unlit selector
    # Flicker the lit set over time in "lights" mode.
    lit_val = lit_h
    if lit_phase != 0.0:
        flick = 0.5 + 0.5 * np.sin(lit_phase * (0.6 + lit_h * 2.0) + wall_h * TAU)
        lit_val = np.clip(lit_h * 0.4 + flick * 0.6, 0.0, 1.0)
    lit = lit_val < lit_frac                          # boolean lit windows

    # ── Cast the interior ray and intersect the room box ──
    dz = 1.0
    dx = lx * persp_eff + pan_x
    dy = ly * persp_eff + pan_y
    # Guard divide-by-zero on axis-aligned rays.
    eps = 1e-6
    dxs = np.where(np.abs(dx) < eps, eps, dx)
    dys = np.where(np.abs(dy) < eps, eps, dy)

    tz = depth / dz                                   # back wall (always +)
    tx = np.where(dx > 0.0, (0.5 - lx) / dxs, (-0.5 - lx) / dxs)
    ty = np.where(dy > 0.0, (0.5 - ly) / dys, (-0.5 - ly) / dys)
    tx = np.where(tx > 0.0, tx, 1e9)
    ty = np.where(ty > 0.0, ty, 1e9)

    t_hit = np.minimum(np.minimum(tz, tx), ty)
    hit_back = (tz <= tx) & (tz <= ty)
    hit_side = (tx < tz) & (tx <= ty)
    hit_ud = (ty < tz) & (ty < tx)                    # floor/ceiling

    # Hit point in room space.
    hx = lx + dx * t_hit
    hy = ly + dy * t_hit
    hz = dz * t_hit                                   # = depth on back wall

    # ── Base surface colour ──
    # Wall hue: warm (beige/amber) vs cool (grey/blue), blended by `warmth`.
    warm = np.array([0.62, 0.50, 0.38])
    cool = np.array([0.40, 0.45, 0.52])
    base = (warm * palette_warm + cool * (1.0 - palette_warm)).reshape(1, 1, 3)
    hue = (0.75 + 0.5 * wall_h)[..., None]
    col = base * hue

    rgb = np.repeat(col, 1, axis=2) * np.ones((h, w, 3))

    # Side walls a touch darker, floor darkest, ceiling lightest — fake AO.
    shade = np.ones((h, w))
    shade = np.where(hit_side, 0.82, shade)
    # up vs down: ceiling (hy>0) lighter, floor (hy<0) darker
    shade = np.where(hit_ud & (hy > 0.0), 1.05, shade)
    shade = np.where(hit_ud & (hy <= 0.0), 0.62, shade)
    rgb = rgb * shade[..., None]

    # ── Back-wall procedural detail (a "picture / furniture" band) ──
    bx = (hx + 0.5)
    by = (hy + 0.5)
    picture = (np.abs(bx - 0.5) < 0.22) & (np.abs(by - 0.55) < 0.16)
    pic_col = np.array([0.20, 0.28, 0.42]).reshape(1, 1, 3)
    back_detail = hit_back[..., None] & picture[..., None]
    rgb = np.where(back_detail, pic_col * np.ones((h, w, 3)), rgb)

    # ── Depth attenuation: deeper surfaces are darker (light falls off) ──
    dnorm = np.clip(hz / (room_depth * 1.3), 0.0, 1.0)
    atten = 1.0 - 0.55 * dnorm
    rgb = rgb * atten[..., None]

    # ── Ceiling light glow for lit rooms ──
    # A soft radial glow near the room's upper-back, only where the window is lit.
    glow_c = np.array([1.0, 0.92, 0.72]).reshape(1, 1, 3)
    gl = np.exp(-(((hx) ** 2) / 0.12 + ((hy - 0.35) ** 2) / 0.10))
    glow = gl * lit.astype(np.float64)
    rgb = rgb + glow_c * (glow[..., None] * 0.9)
    # Lit rooms overall brighter, unlit rooms dim & bluish (moonlight).
    room_bright = np.where(lit, 1.0, 0.30)
    rgb = rgb * room_bright[..., None]
    rgb = np.where((~lit)[..., None],
                   rgb * np.array([0.7, 0.8, 1.0]).reshape(1, 1, 3), rgb)

    # ── Facade glass reflection: faint vertical sky gradient over everything ──
    sky = np.array([0.10, 0.14, 0.22]).reshape(1, 1, 3) * (1.0 - gv)[..., None]
    rgb = rgb + sky * 0.12

    # ── Window frame / mullions (concrete facade) ──
    facade_col = np.array([0.14, 0.14, 0.16]).reshape(1, 1, 3)
    rgb = np.where(in_frame[..., None], facade_col * np.ones((h, w, 3)), rgb)

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    # Field/mask: normalised interior depth (0 at glass, 1 at back wall),
    # zeroed on the facade frame.
    field = np.where(in_frame, 0.0, dnorm).astype(np.float32)
    return rgb, field


@method(
    id="967",
    name="Interior Mapping",
    category="math_art",
    tags=["interior", "facade", "raycast", "procedural", "architecture", "parallax"],
    timeout=300,
    outputs={"image": "IMAGE", "mask": "MASK"},
    params={
        "anim_mode": {
            "description": "animation style (none/pan/lights/zoom)",
            "choices": ["none", "pan", "lights", "zoom"],
            "default": "none",
        },
        "anim_speed": {
            "description": "animation speed multiplier",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "n_cols": {
            "description": "number of window columns across the facade",
            "min": 1.0, "max": 24.0, "default": 8.0,
        },
        "n_rows": {
            "description": "number of window rows down the facade",
            "min": 1.0, "max": 24.0, "default": 6.0,
        },
        "room_depth": {
            "description": "virtual depth of each room (back-wall distance)",
            "min": 0.4, "max": 3.0, "default": 1.4,
        },
        "perspective": {
            "description": "parallax strength (how much the view angle grows off-centre)",
            "min": 0.2, "max": 2.5, "default": 1.1,
        },
        "pan_x": {
            "description": "horizontal virtual-camera offset",
            "min": -1.0, "max": 1.0, "default": 0.0,
        },
        "pan_y": {
            "description": "vertical virtual-camera offset",
            "min": -1.0, "max": 1.0, "default": 0.15,
        },
        "frame_width": {
            "description": "facade mullion / window-frame thickness",
            "min": 0.0, "max": 0.25, "default": 0.06,
        },
        "lit_fraction": {
            "description": "fraction of windows with lights on",
            "min": 0.0, "max": 1.0, "default": 0.6,
        },
        "warmth": {
            "description": "room colour warmth (0=cool grey/blue, 1=warm amber)",
            "min": 0.0, "max": 1.0, "default": 0.5,
        },
        "n_frames": {
            "description": "frames captured for animated modes",
            "min": 8, "max": 240, "default": 60,
        },
        "time": {
            "description": "animation clock (0..2pi) — injected by the executor",
            "min": 0.0, "max": 6.2831853, "default": 0.0,
        },
    },
    inputs={"image_in": "IMAGE"},
)
def method_interior_mapping(out_dir: Path, seed: int, params=None):
    """Interior Mapping — fake 3D building interiors behind a flat facade.

    Per-pixel ray-plane intersection against a virtual room box (van Dongen 2008),
    tiled into a facade of hashed, individually-lit windows. Closed-form
    f(uv, t) — cheap O(W·H) numpy, never hits the render-timeout cull.
    Architecture A internal frame loop.

    A wired IMAGE input modulates overall brightness (Rule 12 override).
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    n_frames = int(params.get("n_frames", 60))

    seed_all(seed)
    params = dict(params)
    params.setdefault("seed", seed)

    _src_lum = wired_source_lum(params, W, H)
    w, h = W, H
    is_anim = anim_mode != "none" or t > 0.01

    def _render(phase: float) -> tuple[np.ndarray, np.ndarray]:
        rgb, field = _compute_interior(w, h, phase, params)
        if _src_lum is not None:
            rgb = np.clip(rgb * (0.4 + 0.6 * _src_lum[..., None]), 0.0, 1.0)
        return rgb, field

    if not is_anim:
        rgb, field = _render(0.0)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
        capture_frame("967", rgb)
        write_scalars(out_dir,
                      mean_luminance=float(rgb.mean()),
                      n_windows=float(int(round(float(params.get("n_cols", 8.0))))
                                      * int(round(float(params.get("n_rows", 6.0))))))
        write_field(out_dir, field)
        write_mask(out_dir, field)
        save(img, mn(967, "Interior Mapping"), out_dir)
        return img

    last_rgb = np.zeros((h, w, 3), dtype=np.float32)
    last_field = np.zeros((h, w), dtype=np.float32)
    for frame in range(n_frames):
        u = frame / max(n_frames - 1, 1)
        phase = t + u * TAU * anim_speed
        rgb, field = _render(phase)
        last_rgb, last_field = rgb, field
        capture_frame("967", rgb)

    img = Image.fromarray((np.clip(last_rgb, 0, 1) * 255).astype(np.uint8), "RGB")
    write_scalars(out_dir, n_frames=float(n_frames),
                  mean_luminance=float(last_rgb.mean()))
    write_field(out_dir, last_field)
    write_mask(out_dir, last_field)
    save(img, mn(967, "Interior Mapping"), out_dir)
    return img
