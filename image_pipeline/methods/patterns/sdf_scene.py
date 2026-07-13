from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame

# ─────────────────────────────────────────────────────────────────────────────
# 2D Signed Distance Field (SDF) Scene — analytic SDF composition + shading.
#
# Technique: real-time SDF rendering (Inigo Quilez, "SDF rendering", 2008-now;
# the workhorse of Shadertoy / procedural real-time graphics). A 2D scene is
# built from *analytic* signed distance functions (circle, box, ring) combined
# with a polynomial smooth-minimum (smin) for organic blends and a domain
# repetition (mod) for tiling, then shaded by the distance field itself: a
# hard edge (smoothstep on d), an exponential outside glow, and distance
# isolines (contour bands). This is qualitatively different from the repo's
# escape-time fractals (Mandelbrot/Julia iterate a recurrence) and from
# metaballs (sum of gaussian falloffs) — here every shape has an *exact*
# signed distance, so blends and contours are crisp and resolution-free.
#
# The render is a pure closed-form f(uv, t): no iteration, no state, so it
# is cheap (vectorised numpy over the whole canvas in <100ms) and sits far
# inside the shootout's 150s render cull. Outputs the RGBA drawing, the raw
# SDF FIELD (normalised), an inside/outside MASK, and SCALAR stats.
# ─────────────────────────────────────────────────────────────────────────────

# Cosmetic ink/paper palettes (color is decoration → color_intrinsic: false,
# so the post-process --recolor pipeline can retint the output freely).
_PALETTES = {
    "amber": (np.array([0.03, 0.02, 0.05]), np.array([0.98, 0.78, 0.36])),
    "ice":   (np.array([0.02, 0.04, 0.06]), np.array([0.55, 0.85, 1.00])),
    "mono":  (np.array([0.00, 0.00, 0.00]), np.array([0.95, 0.95, 0.95])),
}


def _sd_circle(x, y, r):
    return np.sqrt(x * x + y * y) - r


def _sd_box(x, y, bx, by):
    ax = np.abs(x) - bx
    ay = np.abs(y) - by
    ox = np.clip(ax, 0.0, None)
    oy = np.clip(ay, 0.0, None)
    out = np.sqrt(ox * ox + oy * oy) + np.minimum(np.maximum(ax, ay), 0.0)
    return out


def _sd_ring(x, y, r, th):
    return np.abs(np.sqrt(x * x + y * y) - r) - th


def _smin(a, b, k):
    # Polynomial smooth minimum (IQ). k>0 controls blend softness.
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (a * h + b * (1.0 - h)) - k * h * (1.0 - h)


def _scene(px, py, pattern, blend, repetition):
    """Return the signed distance field for the whole canvas.

    px, py: (H, W) float arrays in scene units. Returns (H, W) float SDF.
    """
    rx = px
    ry = py

    # Domain repetition (tiling) — mod folds space into a single cell.
    if repetition > 1e-4:
        rep = max(1e-3, repetition)
        rx = ((rx + 0.5 * rep) % rep) - 0.5 * rep
        ry = ((ry + 0.5 * rep) % rep) - 0.5 * rep

    k = max(1e-3, blend)

    d_circle = _sd_circle(rx, ry, 0.16)
    d_box = _sd_box(rx, ry, 0.20, 0.20)
    d_ring = _sd_ring(rx, ry, 0.34, 0.022)

    if pattern == "blobs":
        d = _smin(d_circle, d_box, k)
    elif pattern == "ring_box":
        d = np.minimum(d_ring, d_box)
    else:  # combo
        d = _smin(_smin(d_circle, d_box, k), d_ring, k)
    return d


def _sdf_render(w, h, seed, params):
    """Pure closed-form SDF render → RGBA float (H, W, 4) in [0, 1].

    No file IO; safe to call from a verification probe with explicit w, h.
    """
    anim_mode = str(params.get("anim_mode", "none"))
    anim_speed = float(params.get("anim_speed", 1.0))
    anim_time = float(params.get("time", 0.0))
    scale = float(params.get("scale", 1.6))
    blend = float(params.get("blend", 0.18))
    repetition = float(params.get("repetition", 0.0))
    glow = float(params.get("glow", 0.8))
    bands = float(params.get("bands", 12.0))
    band_mix = float(params.get("band_mix", 0.5))
    pattern = str(params.get("pattern", "combo"))
    color_mode = str(params.get("color_mode", "amber"))

    # Seed wiring (Step 1): deterministic + per-frame-stable RNG for dither.
    seed_all(seed)
    rng = np.random.default_rng(seed)

    # t-shadowing guard (pitfall #4): never name a loop var `t`.
    _t = 0.0 if anim_mode == "none" else anim_time * anim_speed

    # Animation is applied as a transform of the scene coordinates, so every
    # non-none mode changes the rendered pixels (verified Δ > 0.05).
    angle = _t if anim_mode == "rotate" else 0.0
    ox = 0.30 * math.sin(_t) if anim_mode == "drift" else 0.0
    oy = 0.30 * math.cos(_t * 0.7) if anim_mode == "drift" else 0.0
    glow_eff = glow * (1.0 + 0.6 * math.sin(_t)) if anim_mode == "pulse" else glow
    band_phase = _t if anim_mode in ("rotate", "drift", "pulse") else 0.0

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    aspect = w / max(1, h)
    base_x = (xs / max(1, w - 1) - 0.5) * aspect * 2.0 * scale
    base_y = (ys / max(1, h - 1) - 0.5) * 2.0 * scale

    px = base_x + ox
    py = base_y + oy

    ca, sa = math.cos(angle), math.sin(angle)
    rx = px * ca - py * sa
    ry = px * sa + py * ca

    d = _scene(rx, ry, pattern, blend, repetition)

    # ── Shading from the distance field ──
    bg, ink = _PALETTES.get(color_mode, _PALETTES["amber"])

    edge = 0.014
    inside = np.clip(0.5 - d / (2.0 * edge), 0.0, 1.0)        # 1 inside, 0 outside
    glow_f = np.exp(-3.0 * np.clip(d, 0.0, None)) * glow_eff      # outside halo
    band_f = 0.5 + 0.5 * np.sin(d * max(0.0, bands) - band_phase)  # isolines
    band_factor = (1.0 - band_mix) + band_mix * band_f

    col = bg * (1.0 - inside)[..., None] + ink * inside[..., None]
    col = col + (ink * glow_f[..., None])
    col = col * band_factor[..., None]
    col = np.clip(col, 0.0, 1.0)

    # Subtle film-grain dither (uses the seeded RNG; constant per seed so it
    # does NOT introduce frame-to-frame noise in the static baseline).
    col = col + (rng.random((h, w, 1)).astype(np.float32) - 0.5) / 255.0 * 0.6
    col = np.clip(col, 0.0, 1.0)

    out = np.zeros((h, w, 4), dtype=np.float32)
    out[:, :, 0:3] = col.astype(np.float32)
    out[:, :, 3] = 1.0
    return out, d


@method(
    id="950", name="SDF Scene", category="patterns",
    new_image_contract=True,
    tags=["sdf", "signed-distance", "procedural", "scene", "isoline",
          "glow", "shading", "npr", "animation", "expanded"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "luminance": "SCALAR"},
    params={
        "pattern": {"description": "scene composition (combo blends all 3 shapes; blobs=circle+box; ring_box=ring+box)",
                    "choices": ["combo", "blobs", "ring_box"], "default": "combo"},
        "scale": {"description": "scene zoom / world scale", "min": 0.5, "max": 4.0, "default": 1.6},
        "blend": {"description": "smooth-min blend softness (higher = gooier merge)", "min": 0.01, "max": 0.6, "default": 0.18},
        "repetition": {"description": "domain-repetition cell size (0 = off, tiling on)", "min": 0.0, "max": 1.5, "default": 0.0},
        "glow": {"description": "outside halo strength", "min": 0.0, "max": 2.0, "default": 0.8},
        "bands": {"description": "distance isoline count (contour bands)", "min": 0.0, "max": 40.0, "default": 12.0},
        "band_mix": {"description": "contour-band modulation amount", "min": 0.0, "max": 1.0, "default": 0.5},
        "color_mode": {"description": "cosmetic ink/paper palette", "choices": ["amber", "ice", "mono"], "default": "amber"},
        "anim_mode": {"description": "animation mode: none (static), rotate, drift, pulse",
                      "choices": ["none", "rotate", "drift", "pulse"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation time in radians", "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_sdf_scene(out_dir: Path, seed: int, params=None):
    """2D Signed Distance Field Scene — analytic SDF composition + render.

    Builds a 2D scene from exact signed distance functions (circle, box, ring),
    blends them with a polynomial smooth-minimum and optional domain repetition,
    then shades by the distance field: crisp edges, an exponential outside
    glow, and distance isolines (contour bands). The whole image is a pure
    closed-form f(uv, t) — no iteration, no state — so it is cheap and
    resolution-free.

    Animation modes (deterministic, seed-stable; the SDF is recomputed per
    frame so live preview stays cheap):
      none   - static baseline (identical at every ``time``).
      rotate - the scene spins about its centre (angle = t).
      drift  - the whole scene translates on a Lissajous path.
      pulse  - the glow breathes (1 + 0.6·sin t) and bands sweep.
    """
    try:
        if params is None:
            params = {}
        w = int(W)
        h = int(H)

        out, d = _sdf_render(w, h, seed, params)

        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else float(params.get("time", 0.0)) * anim_speed
        pattern = str(params.get("pattern", "combo"))

        # FIELD: normalised SDF (inside reads bright) — a usable distance map.
        field = np.clip(0.5 - 0.15 * d, 0.0, 1.0).astype(np.float32)
        write_field(out_dir, field)

        # MASK: exact inside/outside selection from raw state (not the image).
        write_mask(out_dir, (d < 0.0).astype(np.float32))

        write_scalars(
            out_dir,
            min_distance=float(float(d.min())),
            coverage=float(float((d < 0.0).mean())),
        )

        capture_frame("950", out)
        save(out, mn(950, f"SDF Scene {pattern} t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 4), dtype=np.float32)
        fallback[:, :, 3] = 1.0
        fallback[:, :, :3] = 0.5
        save(fallback, mn(950, "SDF Scene"), out_dir)
        print(f"[method_950] ERROR: {exc}")
        return fallback
