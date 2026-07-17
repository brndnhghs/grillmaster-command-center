from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    PALETTES,
    load_input,
    write_scalars,
    write_field,
)
from ...core.animation import capture_frame


# ── Value-noise FBM (deterministic, seed-stable) for the procedural height field ──
def _hash_corner(ix, iy, seed):
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    n = (ix * np.uint64(73856093)) ^ (iy * np.uint64(19349663)) ^ (np.uint64(seed) * np.uint64(83492791))
    n = (n ^ (n >> np.uint64(13))) * np.uint64(1274126177)
    n = n ^ (n >> np.uint64(16))
    return (n & np.uint64(0x7FFFFFFF)).astype(np.float64) / 2147483647.0


def _value_noise(x, y, seed):
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    h00 = _hash_corner(xi, yi, seed)
    h10 = _hash_corner(xi + 1, yi, seed)
    h01 = _hash_corner(xi, yi + 1, seed)
    h11 = _hash_corner(xi + 1, yi + 1, seed)
    a = h00 + (h10 - h00) * u
    b = h01 + (h11 - h01) * u
    return (a + (b - a) * v) * 2.0 - 1.0


def _fbm(x, y, seed, octaves=5, lac=2.0, gain=0.5):
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm if norm > 0 else total


def _sample(img, xs, ys):
    """Bilinear sample of img (H,W) or (H,W,C) at float pixel coords, clamped."""
    Hh, Ww = img.shape[:2]
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, Ww - 1)
    x1 = np.clip(x0 + 1, 0, Ww - 1)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, Hh - 1)
    y1 = np.clip(y0 + 1, 0, Hh - 1)
    tx = np.clip(xs - np.floor(xs), 0.0, 1.0)
    ty = np.clip(ys - np.floor(ys), 0.0, 1.0)
    if img.ndim == 2:
        top = img[y0, x0] * (1 - tx) + img[y0, x1] * tx
        bot = img[y1, x0] * (1 - tx) + img[y1, x1] * tx
        return top * (1 - ty) + bot * ty
    top = img[y0, x0] * (1 - tx)[..., None] + img[y0, x1] * tx[..., None]
    bot = img[y1, x0] * (1 - tx)[..., None] + img[y1, x1] * tx[..., None]
    return top * (1 - ty)[..., None] + bot * ty[..., None]


def _procedural_height(Ww, Hh, rng, t, anim_mode, anim_speed):
    """Deterministic height field (rolling fbm) so the relief has structure."""
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    xc = (xx - Ww / 2.0) / max(Ww, Hh)
    yc = (yy - Hh / 2.0) / max(Ww, Hh)
    _t = t * anim_speed
    if anim_mode == "drift":
        xc = xc + 0.1 * _t
        yc = yc + 0.07 * _t
    scale = 3.0 + 2.0 * rng.uniform(0, 1)  # seeded variety
    h = _fbm(xc * scale * 6.0 + (_t if anim_mode == "flow" else 0.0),
             yc * scale * 6.0, int(rng.integers(1, 1 << 30)), octaves=5)
    h = 0.5 + 0.5 * h
    return np.clip(h, 0.0, 1.0)


def _parallax_occlusion(Hh_img, base_color, height, layers, height_scale, light_angle,
                        ambient, lz, Ww_img):
    """Ray-marched parallax occlusion / relief mapping over a 2D height field.

    Marches a view ray (projected to 2D along the light azimuth) through the
    height volume, finds the first self-occluding surface, and shades it by the
    local height-gradient normal under a rotating light. Returns shaded (H,W,3).
    """
    Hh, Ww = height.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    maxd = float(max(Ww, Hh))
    # 2D parallax offset direction (toward the light) — occlusion shifts sampling
    dx = math.cos(light_angle)
    dy = math.sin(light_angle)
    max_off = height_scale * maxd

    step = 1.0 / layers
    cur_h = np.ones((Hh, Ww))
    hit_x = xx.copy()
    hit_y = yy.copy()
    hit_found = np.zeros((Hh, Ww), dtype=bool)

    for _ in range(layers):
        prev_x = hit_x.copy()  # current best guess becomes previous
        prev_y = hit_y.copy()
        prev_h = cur_h.copy()
        cur_h = cur_h - step
        off = max_off * (1.0 - cur_h)
        cur_x = xx + dx * off
        cur_y = yy + dy * off
        h_tex = _sample(height, cur_x, cur_y)
        cond = (cur_h <= h_tex) & (~hit_found)
        if cond.any():
            h_prev_tex = _sample(height, prev_x, prev_y)
            denom = (prev_h - h_prev_tex) + (h_tex - cur_h)
            tt = np.where(denom != 0.0, (prev_h - h_prev_tex) / np.maximum(np.abs(denom), 1e-6), 0.5)
            tt = np.clip(tt, 0.0, 1.0)
            hx = prev_x + tt * (cur_x - prev_x)
            hy = prev_y + tt * (cur_y - prev_y)
            hit_x = np.where(cond, hx, hit_x)
            hit_y = np.where(cond, hy, hit_y)
            hit_found = hit_found | cond

    color = _sample(base_color, hit_x, hit_y)
    if color.ndim == 2:
        color = color[..., None]

    # surface normal from height gradient at the hit point
    hxm = _sample(height, hit_x - 1.0, hit_y)
    hxp = _sample(height, hit_x + 1.0, hit_y)
    hym = _sample(height, hit_x, hit_y - 1.0)
    hyp = _sample(height, hit_x, hit_y + 1.0)
    nx = (hxm - hxp)
    ny = (hym - hyp)
    nz = np.full_like(nx, height_scale * 6.0 + 0.001)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= nlen
    ny /= nlen
    nz /= nlen

    Lx = math.cos(light_angle)
    Ly = math.sin(light_angle)
    diff = np.clip(nx * Lx + ny * Ly + nz * lz, 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * diff
    out = np.clip(color * shade[..., None], 0.0, 1.0)
    return out.astype(np.float32)


@method(
    id="1002",
    name="Parallax Occlusion",
    category="filters",
    tags=["relief", "parallax-occlusion", "height-field", "emboss", "pom", "shading", "animation"],
    params={
        "height_scale": {"description": "relief depth (normalized uv offset)", "min": 0.0, "max": 0.5, "default": 0.15},
        "layers": {"description": "ray-march steps (relief quality)", "min": 8, "max": 64, "default": 32},
        "light_angle": {"description": "light azimuth (radians)", "min": 0.0, "max": 6.2832, "default": 0.7854},
        "ambient": {"description": "ambient shading term", "min": 0.0, "max": 1.0, "default": 0.3},
        "lz": {"description": "light z (relief steepness)", "min": 0.2, "max": 3.0, "default": 1.0},
        "contrast": {"description": "post tone contrast", "min": 0.5, "max": 2.0, "default": 1.0},
        "source": {
            "description": "height field source (wired image luminance, or procedural fbm)",
            "choices": ["procedural", "input_image"],
            "default": "procedural",
        },
        "anim_mode": {
            "description": "animation: none, rotate (light orbits), pulse (relief depth breathes), drift (field scrolls)",
            "choices": ["none", "rotate", "pulse", "drift"],
            "default": "none",
        },
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD"},
    description=(
        "Parallax Occlusion Mapping / Relief Mapping (Policarpo, Oliveira & Comba, "
        "2004/2005; GPU Gems 2 ch. 11). A height field is ray-marched in screen "
        "space: for each pixel a ray is stepped through the height volume along the "
        "view direction, finding the first self-occluding surface. That surface is "
        "shaded by its height-gradient normal under a light, producing a convincing "
        "3D relief from a single 2D image (unlike a plain emboss, it has real "
        "self-occlusion). With a wired image the luminance becomes the height and "
        "the photo is rendered as carved relief; procedural mode builds a rolling "
        "fbm height. Architecture B (closed-form per frame). Cheap (single numpy "
        "march) -> safe for graphs that must dodge the >150s render-timeout cull."
    ),
)
def method_parallax_occlusion(out_dir: Path, seed: int, params=None):
    """Height-field parallax occlusion / relief mapping with rotating-light shading."""
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        height_scale = float(params.get("height_scale", 0.15))
        layers = int(params.get("layers", 32))
        light_angle = float(params.get("light_angle", 0.7854))
        ambient = float(params.get("ambient", 0.3))
        lz = float(params.get("lz", 1.0))
        contrast = float(params.get("contrast", 1.0))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        if anim_mode == "none":
            _t = 0.0
        else:
            _t = t * anim_speed
        if anim_mode == "rotate":
            light_angle = light_angle + _t          # light orbits -> smooth, no cusp
        elif anim_mode == "pulse":
            # Depth breathe: relief swells from a shallow near-flat relief to
            # full depth. Wide range keeps the breath clearly visible (clears
            # the liveness floor at the small default height_scale).
            height_scale = height_scale * (0.1 + 0.9 * (0.5 + 0.5 * math.sin(_t)))
        elif anim_mode == "drift":
            pass  # field scroll handled in _procedural_height

        Hh, Ww = int(H), int(W)

        # ── Build height field + base color (wired input always overrides) ──
        wired = params.get("input_image", "")
        src = None
        if wired:
            try:
                src = load_input(wired, Ww, Hh)
            except (FileNotFoundError, OSError, ValueError):
                src = None
        if src is None and params.get("source", "procedural") == "input_image":
            pass
        if src is None:
            hfield = _procedural_height(Ww, Hh, rng, _t, anim_mode, anim_speed)
            pal = np.array(PALETTES.get("vapor", []), dtype=np.float32) / 255.0
            if pal.shape[0] < 2:
                pal = np.array([[30, 30, 50], [120, 60, 160], [240, 150, 210], [255, 230, 150]],
                                dtype=np.float32) / 255.0
            idx = np.clip((hfield * (pal.shape[0] - 1)).astype(np.int32), 0, pal.shape[0] - 1)
            base_color = pal[idx].astype(np.float64)
        else:
            src = np.asarray(src, dtype=np.float64)
            hfield = (0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2])
            base_color = src

        hfield = np.asarray(hfield, dtype=np.float64)
        base_color = np.asarray(base_color, dtype=np.float64)

        out = _parallax_occlusion(hfield, base_color, hfield, layers, height_scale,
                                  light_angle, ambient, lz, Ww)
        out = np.clip(0.5 + (out - 0.5) * contrast, 0.0, 1.0).astype(np.float32)

        write_field(out_dir, hfield.astype(np.float32))
        write_scalars(
            out_dir,
            height_scale=float(height_scale),
            layers=float(layers),
            light_angle=float(light_angle),
            ambient=float(ambient),
        )
        capture_frame("1002", out)
        save(out, mn(1002, "Parallax Occlusion"), out_dir)
        return out
    except Exception as exc:
        fallback = np.full((int(H), int(W), 3), 128, dtype=np.uint8)
        save(fallback, mn(1002, "Parallax Occlusion"), out_dir)
        print(f"[method_1002] ERROR: {exc}")
        return fallback
