from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES, BG_DEFAULT,
    write_scalars, write_field, write_particles,
)
from ...core.animation import capture_frame


# ─────────────────────────────────────────────────────────────────────────────
# Spirograph — Hypotrochoid / Epitrochoid rosettes.
#
# A Spirograph draws the trace of a point fixed to a small circle of radius r
# rolling inside (hypotrochoid) or outside (epitrochoid) a fixed circle of
# radius R; the pen sits a distance d from the small circle's centre. The
# parametric curve is the classic of mathematical-recreational CG:
#
#   hypotrochoid :  x = (R-r)·cosθ + d·cos(((R-r)/r)·θ)
#                   y = (R-r)·sinθ - d·sin(((R-r)/r)·θ)
#   epitrochoid  :  x = (R+r)·cosθ - d·cos(((R+r)/r)·θ)
#                   y = (R+r)·sinθ - d·sin(((R+r)/r)·θ)
#
# Because the generator is a closed-form function of a single parameter θ, the
# whole pattern is computed in one shot — no simulation state to carry between
# frames, so this is an Architecture-B (per-frame re-call) method whose `time`
# parameter spins, scales, or morphs the rosette. It is deliberately cheap
# (no PDE, no neighbour loops) so it is a healthy shootout seed: it renders in
# well under the 150 s cull threshold that kills ~24 % of genomes.
#
# Reference: B. Dixon, "Mathographics" (1987); the curves are the textbook
# "rose / rosette" family used in generative / recreational CG since the 19th
# century (Cycloid, Spirograph, "Guilloché" engine-turning patterns).
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PARTICLES = 20_000  # cap on written PARTICLES output (disk-friendly)


def _iq_palette(t: np.ndarray, hue_shift: float) -> np.ndarray:
    """Inigo-Quilez cosine palette — smooth, periodic, vivid."""
    t = t + hue_shift
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _color_for(map_name: str, t01: np.ndarray, hue_shift: float,
               pal_name: str) -> np.ndarray:
    """Map parameter-progress t01∈[0,1] to an RGB triple per sample."""
    if map_name == "grayscale":
        g = 0.25 + 0.75 * t01
        return np.stack([g, g, g], axis=-1)
    if map_name == "rainbow":
        return _iq_palette(t01, hue_shift)
    if map_name == "fire":
        return np.stack([
            np.clip(t01 * 1.6, 0.0, 1.0),
            np.clip((t01 - 0.25) * 1.6, 0.0, 1.0),
            np.clip((t01 - 0.6) * 2.2, 0.0, 1.0),
        ], axis=-1)
    if map_name == "ice":
        return np.stack([
            np.clip(t01 * 0.25, 0.0, 1.0),
            np.clip(0.3 + t01 * 0.5, 0.0, 1.0),
            np.clip(0.5 + t01 * 0.5, 0.0, 1.0),
        ], axis=-1)
    if map_name in ("inferno", "viridis", "magma", "plasma", "turbo") and map_name in PALETTES:
        arr = np.asarray(PALETTES[map_name], dtype=np.float32) / 255.0
        idx = np.clip((t01 * (len(arr) - 1)).astype(np.int64), 0, len(arr) - 1)
        return arr[idx]
    if map_name == "palette":
        pal = PALETTES.get(pal_name, PALETTES.get("vapor", PALETTES.get("inferno", [(255, 255, 255)])))
        arr = np.asarray(pal, dtype=np.float32) / 255.0
        idx = np.clip((t01 * (len(arr) - 1)).astype(np.int64), 0, len(arr) - 1)
        return arr[idx]
    # default → IQ ramp
    return _iq_palette(t01, hue_shift)


def _spiro_points(R: int, r: int, d: float, mode: str, theta_max: float,
                  N: int):
    """Return (x, y) arrays of N sample points in abstract units."""
    th = np.linspace(0.0, theta_max, N)
    if mode == "epitrochoid":
        k = (R + r) / float(r)
        x = (R + r) * np.cos(th) - d * np.cos(k * th)
        y = (R + r) * np.sin(th) - d * np.sin(k * th)
    else:  # hypotrochoid
        k = (R - r) / float(r)
        x = (R - r) * np.cos(th) + d * np.cos(k * th)
        y = (R - r) * np.sin(th) - d * np.sin(k * th)
    return x, y


def _closure_theta(R: int, r: int, mode: str) -> float:
    """Exact θ range over which the curve closes (2π · turns)."""
    if r <= 0:
        return 2.0 * math.pi
    turns = r // math.gcd(R, r) if mode == "hypotrochoid" else r // math.gcd(R + r, r)
    turns = max(1, turns)
    return 2.0 * math.pi * turns


@method(
    id="500", name="Spirograph", category="math_art",
    new_image_contract=True,
    tags=["spirograph", "hypotrochoid", "epitrochoid", "rosette", "guilloche",
          "recreational-cg", "procedural", "math-art", "animation", "closed-form"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "luminance": "SCALAR",
             "particles": "PARTICLES"},
    params={
        "R": {"description": "fixed (big) circle radius — number of petals scales with it",
              "min": 3, "max": 60, "default": 35},
        "r": {"description": "rolling (small) circle radius — must be < R for inside rolls",
              "min": 1, "max": 60, "default": 13},
        "d": {"description": "pen distance from the small circle's centre (loop size)",
              "min": 1, "max": 60, "default": 25},
        "mode": {"description": "rolling inside (hypotrochoid) or outside (epitrochoid)",
                 "choices": ["hypotrochoid", "epitrochoid"], "default": "hypotrochoid"},
        "line_width": {"description": "anti-aliased stroke thickness (gaussian sigma, px)",
                       "min": 0.5, "max": 4.0, "default": 1.6},
        "hue_shift": {"description": "hue rotation of the colour ramp along the curve",
                      "min": 0.0, "max": 1.0, "default": 0.0},
        "colormap": {"description": "colour mapping along the curve",
                     "default": "rainbow"},
        "palette": {"description": "palette name for palette mode", "default": "vapor"},
        "bg": {"description": "background mode: neutral grey or dark",
               "choices": ["neutral", "dark"], "default": "neutral"},
        "anim_mode": {"description": "animation mode: none, rotate, breathe, morph",
                      "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
    },
)
def method_spirograph(out_dir, seed: int, params=None):
    """Render a Spirograph rosette (hypotrochoid / epitrochoid) and animate it.

    Closed-form per frame (Architecture B): the orchestrator re-calls this with
    an increasing ``time`` value. In ``none`` mode the output is static.
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        np.random.default_rng(seed)

        R = int(params.get("R", 35))
        r = max(1, int(params.get("r", 13)))
        R = max(r + 1, R) if params.get("mode", "hypotrochoid") == "hypotrochoid" else max(2, R)
        d0 = float(params.get("d", 25))
        mode = params.get("mode", "hypotrochoid")
        line_width = float(params.get("line_width", 1.6))
        hue_shift = float(params.get("hue_shift", 0.0))
        cmap = params.get("colormap", "rainbow")
        pal_name = params.get("palette", "vapor")
        bgmode = params.get("bg", "neutral")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Build the curve (closure guaranteed by exact θ range) ──
        theta_max = _closure_theta(R, r, mode)
        N = min(200_000, max(4000, int(theta_max * 150.0)))
        d_eff = d0
        if anim_mode == "morph":
            # Smooth pen oscillation — no cusp (offset sine). Wide swing so the
            # rosette visibly opens/closes across the animation.
            d_eff = d0 * (0.1 + 0.9 * (0.5 + 0.5 * math.sin(_t * 0.5)))
        x, y = _spiro_points(R, r, d_eff, mode, theta_max, N)

        # ── Fixed reference span (base geometry) ──
        # CRITICAL (pitfall #19): normalize by a FIXED reference span derived
        # from the BASE geometry (d0), never from the animated quantities. If we
        # let d_eff or the post-animation span into the denominator, the breathe
        # scale / morph pen-swing is divided straight back out and the slider
        # does nothing (Δ≈0). With a fixed span, every animated frame is a real
        # size/shape change that reaches the pixels.
        base_span = ((R - r) + d0) if mode == "hypotrochoid" else ((R + r) + d0)

        # ── Animation transforms (continuous, no cusps) ──
        if anim_mode == "rotate":
            ang = _t
            ca, sa = math.cos(ang), math.sin(ang)
            x, y = x * ca - y * sa, x * sa + y * ca
        elif anim_mode == "breathe":
            s = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(_t * 0.5))
            x, y = x * s, y * s

        # ── Map abstract units → centred pixels (fixed span) ──
        span = base_span + 1e-6
        scale = (0.46 * min(W, H)) / span
        cx, cy = W / 2.0, H / 2.0
        px = (x * scale + cx).astype(np.float64)
        py = (y * scale + cy).astype(np.float64)

        # ── Rasterize the polyline (deposit + colour, then gaussian AA) ──
        mask = np.zeros((H, W), dtype=np.float64)
        cr = np.zeros((H, W), dtype=np.float64)
        cg = np.zeros((H, W), dtype=np.float64)
        cb = np.zeros((H, W), dtype=np.float64)

        t01 = np.linspace(0.0, 1.0, N)
        col = _color_for(cmap, t01, hue_shift, pal_name)  # (N, 3)
        xi = np.clip(px.astype(np.int64), 0, W - 1)
        yi = np.clip(py.astype(np.int64), 0, H - 1)
        np.add.at(mask, (yi, xi), 1.0)
        np.add.at(cr, (yi, xi), col[:, 0])
        np.add.at(cg, (yi, xi), col[:, 1])
        np.add.at(cb, (yi, xi), col[:, 2])

        sigma = max(0.5, min(4.0, line_width))
        gmask = gaussian_filter(mask, sigma=sigma)
        gcr = gaussian_filter(cr, sigma=sigma)
        gcg = gaussian_filter(cg, sigma=sigma)
        gcb = gaussian_filter(cb, sigma=sigma)

        cov = gmask / (gmask.max() + 1e-8)
        cov = np.clip(cov, 0.0, 1.0)
        safe = gmask + 1e-8
        linecol = np.stack([gcr / safe, gcg / safe, gcb / safe], axis=-1)

        bg = (np.array(BG_DEFAULT, dtype=np.float64) / 255.0) if bgmode == "neutral" \
            else np.array([0.05, 0.05, 0.07], dtype=np.float64)
        rgb = bg * (1.0 - cov)[..., None] + linecol * cov[..., None]
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Auxiliary outputs ──
        write_field(out_dir, cov.astype(np.float32))
        write_scalars(out_dir, points=float(N), petals=float(theta_max / (2.0 * math.pi)),
                      coverage=float(cov.mean()))
        # PARTICLES: stride the vertices to the cap
        stride = max(1, N // _MAX_PARTICLES)
        vp = np.stack([px[::stride], py[::stride]], axis=-1)
        vx = np.roll(px, -1)[::stride] - vp[:, 0]
        vy = np.roll(py, -1)[::stride] - vp[:, 1]
        parts = np.stack([vp[:, 0], vp[:, 1], vx, vy], axis=-1).astype(np.float32)
        write_particles(out_dir, parts)

        # ── Save (Architecture B: name carries time so frames don't collide) ──
        capture_frame("500", rgb)
        save(rgb, mn(500, f"Spirograph t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 128, dtype=np.uint8)
        save(fallback, mn(500, "Spirograph"), out_dir)
        print(f"[method_500] ERROR: {exc}")
        return fallback
