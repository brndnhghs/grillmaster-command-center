from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save,
    norm,
    mn,
    seed_all,
    BG_DEFAULT,
    W,
    H,
    write_field,
    write_mask,
    write_scalars,
)
from ...core.animation import capture_frame


def _value_noise_2d(shape, rng, scale=8):
    """Simple tileable value noise via a coarse random grid + bilinear upsample."""
    hh, ww = shape
    gh = max(2, hh // scale)
    gw = max(2, ww // scale)
    grid = rng.random((gh + 1, gw + 1)).astype(np.float32)
    ys = np.linspace(0, gh, hh).astype(np.float32)
    xs = np.linspace(0, gw, ww).astype(np.float32)
    y0 = np.floor(ys).astype(int) % gh
    y1 = (y0 + 1) % gh
    x0 = np.floor(xs).astype(int) % gw
    x1 = (x0 + 1) % gw
    fy = (ys - np.floor(ys))[:, None]
    fx = (xs - np.floor(xs))[None, :]
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx
    bot = grid[y1][:, x0] * (1 - fx) + grid[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def _hsv2rgb(h, s, v):
    """Vectorized HSV→RGB. h,v are arrays; s is a scalar."""
    h = h % 1.0
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


@method(
    id="923",
    name="MatCap Relight",
    category="filters",
    new_image_contract=True,
    tags=["matcap", "relight", "shading", "npr", "animation"],
    description=(
        "Material-Capture (MatCap) lit-sphere relighting of a height/intensity "
        "field. Estimates a surface normal from a height source (wired upstream "
        "image luminance, or a generated sphere/bumps/noise) and shades it with a "
        "procedurally generated MatCap — clay / pearl / chrome / iridescent — no "
        "texture needed."
    ),
    params={
        "source": {"description": "height source when no upstream image is wired (sphere/bumps/noise/checker/gradient)", "default": "sphere"},
        "matcap": {"description": "material style (clay/pearl/chrome/iridescent)", "choices": ["clay", "pearl", "chrome", "iridescent"], "default": "pearl"},
        "light_dir": {"description": "key light azimuth (radians)", "min": 0.0, "max": 6.2832, "default": 0.6},
        "relief": {"description": "surface relief / depth-from-luminance strength", "min": 0.0, "max": 4.0, "default": 1.0},
        "strength": {"description": "shading mix toward matcap (0=flat gray, 1=full matcap)", "min": 0.0, "max": 1.0, "default": 1.0},
        "spec_pow": {"description": "specular exponent", "min": 2.0, "max": 128.0, "default": 24.0},
        "albedo_r": {"description": "base albedo red", "min": 0.0, "max": 1.0, "default": 0.85},
        "albedo_g": {"description": "base albedo green", "min": 0.0, "max": 1.0, "default": 0.55},
        "albedo_b": {"description": "base albedo blue", "min": 0.0, "max": 1.0, "default": 0.35},
        "anim_mode": {"description": "animation mode (none/light_orbit/sphere_spin/relief_breathe)", "choices": ["none", "light_orbit", "sphere_spin", "relief_breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time (radians)", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
    outputs={"image": "IMAGE", "luminance": "FIELD", "field": "FIELD", "mask": "MASK"},
)
def method_matcap_relight(out_dir: Path, seed: int, params=None):
    """MatCap Relight — 2.5D material-capture shading of a height/intensity field.

    Estimates a surface normal from a height field (wired upstream image
    luminance, or a generated source), then shades it with a procedurally
    generated Material-Capture (MatCap) lit sphere — no texture needed. The
    MatCap technique (Blinn & Newell 1976; revived 2023 by neural-matcap
    models such as MatFuse, arXiv:2308.11408) turns a flat image into a
    glossy / pearly / chrome 3D-looking render via a single normal lookup.

    Parameters:
        source (str): height source when unwired (sphere/bumps/noise/checker/gradient)
        matcap (str): material style (clay/pearl/chrome/iridescent)
        light_dir (float): key light azimuth (0-2pi)
        relief (float): surface relief / depth-from-luminance (0-4)
        strength (float): mix toward matcap (0-1)
        spec_pow (float): specular exponent (2-128)
        albedo_r/g/b (float): base albedo color (0-1)
        anim_mode (str): none/light_orbit/sphere_spin/relief_breathe
        anim_speed (float): animation speed (0-5)
        time (float): animation time (0-2pi)
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    src = str(params.get("source", "sphere"))
    matcap = str(params.get("matcap", "pearl"))
    light_dir = float(params.get("light_dir", 0.6))
    relief = float(params.get("relief", 1.0))
    strength = float(params.get("strength", 1.0))
    spec_pow = float(params.get("spec_pow", 24.0))
    alb = np.array(
        [
            float(params.get("albedo_r", 0.85)),
            float(params.get("albedo_g", 0.55)),
            float(params.get("albedo_b", 0.35)),
        ],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    # ── Animation modulation (no t-shadowing: t is kept a scalar clock) ──
    if anim_mode == "light_orbit":
        light_dir = (light_dir + t) % (2.0 * math.pi)
    elif anim_mode == "relief_breathe":
        # Smooth breathing of surface relief (cos-based, no abs() cusps). Wide
        # 0.15..1.0 swing keeps the effect clearly visible (dead-control guard).
        relief = relief * (0.15 + 0.85 * (0.5 - 0.5 * math.cos(t)))
    spin = t if anim_mode == "sphere_spin" else 0.0

    hh, ww = int(H), int(W)
    yy, xx = np.mgrid[:hh, :ww].astype(np.float32)
    cx, cy = ww / 2.0, hh / 2.0
    max_r = math.sqrt(cx * cx + cy * cy) + 1e-6

    # ── Wired input override (Architecture: new_image_contract passes the
    #    upstream image in-memory as params["_input_image"], H×W×3 float [0,1]).
    #    Resize defensively so any wired size maps to the canvas.) ──
    wired = params.get("_input_image", None)
    if wired is not None:
        himg = np.asarray(wired, dtype=np.float32)
        if himg.ndim == 3 and himg.shape[2] >= 3:
            himg = himg[..., :3]
        elif himg.ndim == 2:
            himg = himg[..., None].repeat(3, axis=-1)
        if himg.shape[0] != hh or himg.shape[1] != ww:
            from PIL import Image as _PIL

            himg = np.array(
                _PIL.fromarray((np.clip(himg, 0.0, 1.0) * 255).astype(np.uint8)).resize((ww, hh))
            ).astype(np.float32) / 255.0
        h = norm(np.mean(himg[..., :3], axis=-1))
    else:
        if src == "sphere":
            u = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_r
            h = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))  # true hemisphere
        elif src == "bumps":
            g1 = np.exp(-(((xx - cx * 0.6) ** 2 + (yy - cy * 0.6) ** 2) / (2 * (min(cx, cy) * 0.18) ** 2)))
            g2 = np.exp(-(((xx - cx * 1.4) ** 2 + (yy - cy * 1.4) ** 2) / (2 * (min(cx, cy) * 0.12) ** 2)))
            h = norm(g1 + 0.7 * g2 + 0.3 * _value_noise_2d((hh, ww), rng, scale=6))
        elif src == "noise":
            h = norm(
                _value_noise_2d((hh, ww), rng, scale=10) * 0.6
                + _value_noise_2d((hh, ww), rng, scale=4) * 0.4
            )
        elif src == "checker":
            h = ((xx // 40 + yy // 40) % 2).astype(np.float32)
        elif src == "gradient":
            h = norm(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2))
        else:
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_r
            h = np.clip(1.0 - r, 0.0, 1.0)

    # ── Normal from height (central differences, smooth, no abs() cusps) ──
    gx = np.zeros((hh, ww), dtype=np.float32)
    gy = np.zeros((hh, ww), dtype=np.float32)
    gx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    gy[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
    gx[:, 0] = h[:, 1] - h[:, 0]
    gx[:, -1] = h[:, -1] - h[:, -2]
    gy[0, :] = h[1, :] - h[0, :]
    gy[-1, :] = h[-1, :] - h[-2, :]

    # Bump-map normal: scale the gradient by the canvas height so a feature
    # spanning the whole canvas yields a ~45deg slope at relief=1. Height fields
    # live in [0,1] over a multi-hundred-px canvas, so raw per-pixel gradients
    # are tiny and would make lighting/animation inert (dead control). The
    # `relief` slider then modulates this gain live.
    GAIN = float(H)
    slope = relief * GAIN
    n = np.empty((hh, ww, 3), dtype=np.float32)
    n[..., 0] = -gx * slope
    n[..., 1] = -gy * slope
    n[..., 2] = 1.0
    mag = np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2)
    MAXS = 6.0  # clamp tilt to ~80deg so very steep regions stay valid
    big = mag > MAXS
    if np.any(big):
        s = np.where(big, MAXS / (mag + 1e-6), 1.0)
        n[..., 0] = n[..., 0] * s
        n[..., 1] = n[..., 1] * s
    nlen = np.sqrt((n ** 2).sum(axis=-1, keepdims=True)) + 1e-6
    n = n / nlen

    # sphere_spin: rotate the normal's xy about the view axis (no symmetry-cancel)
    if spin != 0.0:
        cs, sn = math.cos(spin), math.sin(spin)
        nx, ny = n[..., 0].copy(), n[..., 1].copy()
        n[..., 0] = nx * cs - ny * sn
        n[..., 1] = nx * sn + ny * cs

    # ── Procedural MatCap shading (no texture) ──
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    L = np.array([math.cos(light_dir), math.sin(light_dir), 0.7], dtype=np.float32)
    L = L / (np.linalg.norm(L) + 1e-6)
    half = L + V
    half = half / (np.linalg.norm(half) + 1e-6)

    ndl = np.clip((n * L).sum(axis=-1), 0.0, 1.0)
    ndh = np.clip((n * half).sum(axis=-1), 0.0, 1.0)
    fres = np.clip(1.0 - n[..., 2], 0.0, 1.0) ** 3
    spec = ndh ** spec_pow

    ndl3 = ndl[..., None]
    fres3 = fres[..., None]
    spec3 = spec[..., None]

    if matcap == "clay":
        out = alb * (0.25 + 0.75 * ndl3) + 0.12 * fres3 * np.array([1.0, 0.9, 0.8], dtype=np.float32).reshape(1, 1, 3)
    elif matcap == "pearl":
        rim = np.array([0.9, 0.8, 1.0], dtype=np.float32).reshape(1, 1, 3)
        out = alb * (0.3 + 0.7 * ndl3) + 0.5 * spec3 + 0.25 * fres3 * rim
    elif matcap == "chrome":
        sky = np.array([0.6, 0.75, 0.95], dtype=np.float32).reshape(1, 1, 3)
        ground = np.array([0.12, 0.12, 0.14], dtype=np.float32).reshape(1, 1, 3)
        horizon = np.array([0.95, 0.92, 0.88], dtype=np.float32).reshape(1, 1, 3)
        ty = np.clip(n[..., 1] * 0.5 + 0.5, 0.0, 1.0)
        flo = (ty * 2.0)[..., None]
        fhi = ((ty - 0.5) * 2.0)[..., None]
        lo = (ty < 0.5)[..., None]
        env = np.where(
            lo,
            ground * (1.0 - flo) + horizon * flo,
            horizon * (1.0 - fhi) + sky * fhi,
        )
        out = env * (0.45 + 0.55 * ndl3) + 1.2 * spec3
    elif matcap == "iridescent":
        ang = np.arctan2(n[..., 1], n[..., 0]) / (2.0 * math.pi) + 0.5 + 0.2 * ndl
        val = np.clip(0.4 + 0.6 * ndl, 0.0, 1.0)
        out = _hsv2rgb(ang % 1.0, 0.6, val) + 0.5 * spec3
    else:
        out = alb * (0.3 + 0.7 * ndl3)

    out = out * strength + (1.0 - strength) * 0.5
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    # ── Slope mask (steep regions = feature/edge selection) ──
    slope = np.sqrt(gx ** 2 + gy ** 2) * relief
    mask = np.clip(norm(slope), 0.0, 1.0)

    try:
        capture_frame("923", out)
        save(out, mn(923, "MatCap Relight"), out_dir)
        write_field(out_dir, h.astype(np.float32))
        write_mask(out_dir, mask.astype(np.float32))
        write_scalars(out_dir, light_dir=float(light_dir), relief=float(relief), mean_slope=float(slope.mean()))
    except Exception:
        # Rule 1: PNG in every code path
        fallback = np.zeros((hh, ww, 3), dtype=np.float32) + np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        save(fallback, mn(923, "MatCap Relight"), out_dir)
        raise

