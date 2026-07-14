from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.animation import capture_frame
from ...core.utils import save, mn, W, H, write_field, write_scalars, seed_all


# ─── IFS presets ────────────────────────────────────────────────────────────
# Each preset is a list of (a, b, c, d, e, f, prob) affine maps
#   x' = a*x + b*y + c
#   y' = d*x + e*y + f
# and a `rotate` flag marking maps whose linear part should be spun (rarely off).
_PRESETS = {
    "sierpinski": [
        (0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 1 / 3),
        (0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 1 / 3),
        (0.5, 0.0, 0.25, 0.0, 0.5, 0.4330127, 1 / 3),
    ],
    # Heighway dragon: two maps, each a 45°/135° rotation scaled by 1/√2.
    "dragon": [
        (0.5, -0.5, 0.0, 0.5, 0.5, 0.0, 0.5),
        (-0.5, -0.5, 1.0, 0.5, -0.5, 0.0, 0.5),
    ],
    # Sierpinski carpet: 8 of the 9 sub-squares (scale 1/3).
    "carpet": [
        (1 / 3, 0.0, 0.0, 0.0, 1 / 3, 0.0, 1 / 8),
        (1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 8),
        (1 / 3, 0.0, 2 / 3, 0.0, 1 / 3, 0.0, 1 / 8),
        (1 / 3, 0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 8),
        (1 / 3, 0.0, 2 / 3, 0.0, 1 / 3, 1 / 3, 1 / 8),
        (1 / 3, 0.0, 0.0, 0.0, 1 / 3, 2 / 3, 1 / 8),
        (1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 2 / 3, 1 / 8),
        (1 / 3, 0.0, 2 / 3, 0.0, 1 / 3, 2 / 3, 1 / 8),
    ],
    # Koch-like snowflake: 6 maps, each a 60° rotation scaled by 1/3.
    "snowflake": [
        (0.5, -0.288675, 0.3333, 0.288675, 0.5, 0.0, 1 / 6),
        (0.5, -0.288675, 0.3333, 0.288675, 0.5, 0.0, 1 / 6),  # placeholder, overwritten below
    ],
    # Spiral galaxy: two counter-rotating contractive maps.
    "spiral": [
        (0.9 * math.cos(0.35), -0.9 * math.sin(0.35), 0.0,
         0.9 * math.sin(0.35), 0.9 * math.cos(0.35), 0.0, 0.5),
        (0.9 * math.cos(-0.35), -0.9 * math.sin(-0.35), 0.45,
         0.9 * math.sin(-0.35), 0.9 * math.cos(-0.35), 0.45, 0.5),
    ],
}
# Build the snowflake properly (6 maps at 60° increments, scale 1/3).
_sf = []
for _k in range(6):
    _ang = _k * math.pi / 3.0
    _cx, _sx = math.cos(_ang), math.sin(_ang)
    _s = 1.0 / 3.0
    _sf.append((_s * _cx, -_s * _sx, 0.5, _s * _sx, _s * _cx, 0.5, 1 / 6))
_PRESETS["snowflake"] = _sf


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def _chaos_game(maps, probs, n, rng):
    """Classic chaos game: iterate n times, applying a randomly chosen affine
    map each step. Returns (xs, ys, tone) arrays of kept points."""
    nm = len(maps)
    A = np.array([[m[0], m[1]] for m in maps], dtype=np.float64)
    B = np.array([[m[2], m[3]] for m in maps], dtype=np.float64)  # c, d
    C = np.array([[m[4], m[5]] for m in maps], dtype=np.float64)   # e, f
    choices = rng.choice(nm, size=n, p=probs)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    tone = np.empty(n, dtype=np.float32)
    kidx = np.empty(n, dtype=np.int32)
    x = 0.0
    y = 0.0
    skip = 24
    for i in range(n):
        k = int(choices[i])
        nx = A[k, 0] * x + A[k, 1] * y + B[k, 0]          # a*x + b*y + c
        ny = B[k, 1] * x + C[k, 0] * y + C[k, 1]          # d*x + e*y + f
        x, y = nx, ny
        if i >= skip:
            j = i - skip
            xs[j] = x
            ys[j] = y
            tone[j] = i / n
            kidx[j] = k
    return xs, ys, tone, kidx


def _render(xs, ys, tone, kidx, coloring, hue_shift, n_maps):
    hh, ww = H, W
    # Fit points into the canvas (centred, uniform scale, margin).
    margin = 0.06
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    sw = (ww * (1 - 2 * margin)) / max(1e-9, xmax - xmin)
    sh = (hh * (1 - 2 * margin)) / max(1e-9, ymax - ymin)
    s = min(sw, sh)
    px = (xs - xmin) * s + margin * ww + (ww - ((xmax - xmin) * s + 2 * margin * ww)) / 2.0
    py = (ys - ymin) * s + margin * hh + (hh - ((ymax - ymin) * s + 2 * margin * hh)) / 2.0

    # Per-point colour.
    if coloring == "transform":
        tone = kidx.astype(np.float32) / max(1, n_maps - 1)
    elif coloring == "solid":
        tone = np.full(len(xs), 0.55, dtype=np.float32)
    hue = (tone + hue_shift) % 1.0
    sat = 0.72
    val = 1.0
    cols = np.empty((len(xs), 3), dtype=np.float64)
    for idx in range(len(xs)):
        r, g, b = hsv_to_rgb(float(hue[idx]), sat, val)
        cols[idx, 0] = r
        cols[idx, 1] = g
        cols[idx, 2] = b

    dens, _, _ = np.histogram2d(py, px, bins=[hh, ww], range=[[0, hh], [0, ww]])
    sum_r, _, _ = np.histogram2d(py, px, bins=[hh, ww], range=[[0, hh], [0, ww]], weights=cols[:, 0])
    sum_g, _, _ = np.histogram2d(py, px, bins=[hh, ww], range=[[0, hh], [0, ww]], weights=cols[:, 1])
    sum_b, _, _ = np.histogram2d(py, px, bins=[hh, ww], range=[[0, hh], [0, ww]], weights=cols[:, 2])
    dens = dens.astype(np.float32)
    sum_r = sum_r.astype(np.float32)
    sum_g = sum_g.astype(np.float32)
    sum_b = sum_b.astype(np.float32)

    maxd = dens.max()
    if maxd <= 0:
        return np.zeros((hh, ww, 3), dtype=np.float32), dens
    bright = np.log1p(dens) / np.log1p(maxd)  # log-density tone mapping
    inv = 1.0 / (dens + 1e-9)
    cr = (sum_r * inv).clip(0, 1)
    cg = (sum_g * inv).clip(0, 1)
    cb = (sum_b * inv).clip(0, 1)
    out = np.stack([cr, cg, cb], -1) * bright[..., None]
    return out.astype(np.float32), dens.astype(np.float32)


@method(
    id="353",
    name="IFS Fractal",
    category="patterns",
    new_image_contract=True,
    tags=["fractal", "ifs", "chaos-game", "generative", "self-similar", "animation"],
    inputs={},
    outputs={"image": "IMAGE"},
    params={
        "preset": {"description": "IFS preset (sierpinski/dragon/carpet/snowflake/spiral)", "choices": ["sierpinski", "dragon", "carpet", "snowflake", "spiral"], "default": "sierpinski"},
        "points": {"description": "chaos-game iterations (more = denser)", "min": 10000, "max": 500000, "default": 120000},
        "coloring": {"description": "how points are coloured", "choices": ["iteration", "transform", "solid"], "default": "iteration"},
        "hue_shift": {"description": "rotate the colour ramp", "min": 0.0, "max": 1.0, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/spin)", "choices": ["none", "spin"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_ifs_fractal(out_dir: Path, seed: int, params=None):
    """IFS Fractal — self-similar shapes via the Chaos Game (Barnsley 1988).

    An iterated function system (IFS) is a finite set of contractive affine
    maps on the plane. By the contraction mapping theorem their union has a
    unique non-empty fixed point — the attractor — which is exactly the fractal
    we see. The Chaos Game (Barnsley & Sloan 1988) is the standard way to *draw*
    it: start anywhere, repeatedly pick a map at random (weighted by its
    probability) and jump to its image. After a short transient the visited
    points carpet the attractor, no matter the start.

        x' = a·x + b·y + c
        y' = d·x + e·y + f

    Rendering uses the fractal-flame log-density trick (Draves 1993): each pixel
    accumulates point count *and* a colour sum, brightness = log(1+count) /
    log(1+max), and the per-pixel colour is the density-weighted average. This
    gives the soft, glowing look that linear density cannot.

    Presets:
      sierpinski — the triangle (3 half-scale maps)
      dragon     — the Heighway dragon (two √½ rotations)
      carpet     — Sierpinski carpet (8 / 9 sub-squares)
      snowflake  — 6-fold Koch-style snowflake
      spiral     — two counter-rotating contractive maps (a spiral galaxy)

    Colouring: `iteration` ramps hue along the draw order, `transform` colours by
    which map produced each point, `solid` uses one colour.

    Animation (spin): the generated point cloud is rigidly rotated and gently
    non-uniformly scaled about the canvas centre — a smooth, cusp-free "breathing
    spin" that reads clearly at every frame (verified by changed-pixel fraction,
    not mean-Δ, since fractals like Sierpinski are rotationally symmetric).

    This CPU path is the authoritative export; it is a pure function of the seed
    and params (deterministic).
    """
    if params is None:
        params = {}
    seed_all(seed)
    rng = np.random.default_rng(seed)

    preset = str(params.get("preset", "sierpinski"))
    n_points = int(params.get("points", 120000))
    coloring = str(params.get("coloring", "iteration"))
    hue_shift = float(params.get("hue_shift", 0.0))
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = float(params.get("time", 0.0)) * anim_speed

    n_points = max(10000, min(500000, n_points))
    if preset not in _PRESETS:
        preset = "sierpinski"
    maps = _PRESETS[preset]
    probs = np.array([m[6] for m in maps], dtype=np.float64)
    probs = probs / probs.sum()

    xs, ys, tone, kidx = _chaos_game(maps, probs, n_points, rng)

    # Spin / deform transform about the canvas centre.
    if anim_mode == "spin":
        ang = 0.15 * math.sin(t)
        sx = 1.0 + 0.08 * math.sin(t)
        sy = 1.0 - 0.08 * math.sin(t)
        ca, sa = math.cos(ang), math.sin(ang)
        cx, cy = W / 2.0, H / 2.0
        # work in pixel space after a first fit is done inside _render; apply
        # here by rotating the normalised point cloud before fitting instead.
        # Rotate the world points about their centroid, then let _render fit.
        mxc, myc = xs.mean(), ys.mean()
        dx = xs - mxc
        dy = ys - myc
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
        xs = mxc + rx * sx
        ys = myc + ry * sy

    out, dens = _render(xs, ys, tone, kidx, coloring, hue_shift, len(maps))

    fill = float((dens > 0).mean())
    write_scalars(out_dir, preset=float(hash(preset) & 0xffff), points=float(n_points),
                  coloring=float(hash(coloring) & 0xffff), fill=fill)
    # Density grid is a meaningful 2D field (potential map for downstream wiring).
    write_field(out_dir, dens / max(1e-9, dens.max()))

    capture_frame("353", out)
    try:
        save(out, mn(353, f"IFS {preset} ({coloring})"), out_dir)
    except Exception:
        save(out, mn(353, "IFS Fractal"), out_dir)
