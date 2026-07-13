from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, PALETTES,
    write_scalars, write_field, write_particles,
)
from ...core.animation import capture_frame


# ─────────────────────────────────────────────────────────────────────────────
# De Jong strange attractor (Peter de Jong, 1987; popularised by J. C. Sprott).
#
# The 2D discrete map
#     x_{n+1} = sin(a·y_n) - cos(b·x_n)
#     y_{n+1} = sin(c·x_n) - cos(d·y_n)
# is a paragon of "simple map, complex dynamics": for almost any (a,b,c,d) the
# orbit densely fills a fractal-shaped subset of [-2, 2]². Plotting millions of
# iterated points and accumulating their density yields the luminous, smoke-like
# clouds that are a staple of generative / mathematical CG art.
#
# Reference: P. de Jong, "The Science Game" (1987); J. C. Sprott, "Strange
# Attractors: Creating Patterns in Chaos" (2003). The map is the canonical
# example of a *strange attractor* produced by a very short closed-form recurre
# — no simulation state to carry between frames, so it is an Architecture-B
# (per-frame re-call) method whose `time` parameter morphs the parameters.
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PARTICLES = 200_000  # cap on written PARTICLES output (disk-friendly)


def _iq_ramp(t: np.ndarray) -> np.ndarray:
    """Inigo-Quilez cosine palette — smooth, periodic, vivid."""
    t = np.clip(t, 0.0, 1.0)
    r = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.00))
    g = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.3333333))
    b = 0.5 + 0.5 * np.cos(6.2831853 * (t + 0.6666667))
    return np.stack([r, g, b], axis=-1)


def _inferno(t: np.ndarray) -> np.ndarray:
    """Inferno colormap; use PALETTES if present, else an IQ warm fallback."""
    pal = PALETTES.get("inferno", [])
    if len(pal) >= 2:
        arr = np.asarray(pal, dtype=np.float32) / 255.0
        idx = np.clip((t * (len(arr) - 1)).astype(np.int64), 0, len(arr) - 1)
        return arr[idx]
    # Warm IQ fallback approximating inferno's black→red→yellow→white ramp.
    r = np.clip(t * 1.6, 0.0, 1.0)
    g = np.clip((t - 0.25) * 1.6, 0.0, 1.0)
    b = np.clip((t - 0.6) * 2.2, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


@method(
    id="498", name="De Jong Attractor", category="math_art",
    new_image_contract=True,
    tags=["attractor", "de-jong", "chaos", "strange-attractor", "procedural",
          "generative", "math-art", "animation", "expanded"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "luminance": "SCALAR",
             "particles": "PARTICLES"},
    params={
        "a": {"description": "de Jong parameter a (shape control)",
              "min": -3.0, "max": 3.0, "default": -2.0},
        "b": {"description": "de Jong parameter b (shape control)",
              "min": -3.0, "max": 3.0, "default": -2.0},
        "c": {"description": "de Jong parameter c (shape control)",
              "min": -3.0, "max": 3.0, "default": -1.2},
        "d": {"description": "de Jong parameter d (shape control)",
              "min": -3.0, "max": 3.0, "default": 2.0},
        "walkers": {"description": "parallel trajectories; each samples `steps` points",
                    "min": 200, "max": 8000, "default": 2000},
        "steps": {"description": "points sampled per walker (after transient discard)",
                  "min": 200, "max": 4000, "default": 1000},
        "discard": {"description": "initial transient steps discarded per walker",
                    "min": 0, "max": 500, "default": 100},
        "color_mode": {"description": "coloring: density (inferno), velocity (flow hue), fire, ice, mono",
                       "choices": ["density", "velocity", "fire", "ice", "mono"],
                       "default": "density"},
        "exposure": {"description": "tone-map exposure (higher = brighter / denser glow)",
                     "min": 0.2, "max": 6.0, "default": 1.6},
        "background": {"description": "canvas background",
                       "choices": ["black", "navy", "cream", "white"], "default": "black"},
        "anim_mode": {"description": "animation mode: none (static), morph_a, morph_all, orbit",
                      "choices": ["none", "morph_a", "morph_all", "orbit"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier",
                       "min": 0.1, "max": 5.0, "default": 1.0},
        "time": {"description": "animation phase [0, 2π)",
                 "min": 0.0, "max": 6.2832, "default": 0.0},
    },
)
def method_de_jong(out_dir: Path, seed: int, params=None):
    """De Jong strange attractor — a closed-form chaos map rendered as a density cloud.

    Iterates the de Jong recurrence x' = sin(a·y) − cos(b·x), y' = sin(c·x) −
    cos(d·y) from many parallel starting points, splats every visited point into
    a density grid, and tone-maps the accumulation so the attractor's fractal
    skeleton glows. Coloring options: ``density`` (inferno by local density),
    ``velocity`` (hue from the local flow direction), ``fire`` / ``ice`` /
    ``mono`` ramps.

    Because the map is a pure function of its four parameters, it is an
    Architecture-B method: the orchestrator re-calls it with an increasing
    ``time`` value, and ``anim_mode`` perturbs the parameters smoothly (via sine/
    cosine of ``t`` — no ``abs(sin)`` cusps) so the attractor morphs frame to
    frame.

    Pipeline (per frame):
      1. Seed `walkers` independent start points (deterministic from `seed`).
      2. Discard `discard` transient steps so each walker is on the attractor.
      3. For `steps` iterations, apply the map in lock-step over all walkers
         (fully vectorised) and record every (x, y) plus its step velocity.
      4. Fit the point cloud to the canvas and splat into a density grid +
         per-pixel accumulated velocity vectors.
      5. Tone-map density (1 − exp(−exposure·ρ)) and composite the chosen
         coloring over the background.
      6. Emit FIELD = normalised density, PARTICLES = a capped sample of the
         visited points, SCALAR = parameter/coverage stats.

    Determinism: all RNG is seeded by `seed`; in ``none`` mode the output is
    identical at every `time` (Δ ≈ 0 static baseline).
    """
    try:
        if params is None:
            params = {}

        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        a = float(params.get("a", -2.0))
        b = float(params.get("b", -2.0))
        c = float(params.get("c", -1.2))
        d = float(params.get("d", 2.0))
        walkers = max(200, min(8000, int(params.get("walkers", 2000))))
        steps = max(200, min(4000, int(params.get("steps", 1000))))
        discard = max(0, min(500, int(params.get("discard", 100))))
        color_mode = str(params.get("color_mode", "density"))
        exposure = max(0.2, min(6.0, float(params.get("exposure", 1.6))))
        background = str(params.get("background", "black"))

        # ── Animation: smooth parameter perturbation (no cusps) ──
        if anim_mode == "morph_a":
            a = a + 1.2 * math.sin(_t)
        elif anim_mode == "morph_all":
            a = a + 0.9 * math.sin(_t)
            b = b + 0.9 * math.cos(_t * 0.9)
            c = c + 0.9 * math.sin(_t * 1.1 + 1.0)
            d = d + 0.9 * math.cos(_t * 0.8 + 2.0)
        elif anim_mode == "orbit":
            a = a + 0.8 * math.cos(_t)
            b = b + 0.8 * math.sin(_t)

        w = int(W)
        h = int(H)

        seed_all(seed)
        rng = np.random.default_rng(seed)

        # Parallel walkers: each is an independent trajectory; vectorising over
        # walkers gives millions of attractor points with O(walkers) numpy ops.
        xs = rng.uniform(-1.0, 1.0, size=walkers)
        ys = rng.uniform(-1.0, 1.0, size=walkers)

        # Burn transient steps so every walker has settled onto the attractor.
        for _ in range(discard):
            nx = np.sin(a * ys) - np.cos(b * xs)
            ny = np.sin(c * xs) - np.cos(d * ys)
            xs, ys = nx, ny

        total = walkers * steps
        px = np.empty(total, dtype=np.float32)
        py = np.empty(total, dtype=np.float32)
        pvx = np.empty(total, dtype=np.float32)
        pvy = np.empty(total, dtype=np.float32)

        off = 0
        for _s in range(steps):
            nx = np.sin(a * ys) - np.cos(b * xs)
            ny = np.sin(c * xs) - np.cos(d * ys)
            vx = nx - xs
            vy = ny - ys
            seg = slice(off, off + walkers)
            px[seg] = xs.astype(np.float32)
            py[seg] = ys.astype(np.float32)
            pvx[seg] = vx.astype(np.float32)
            pvy[seg] = vy.astype(np.float32)
            xs, ys = nx, ny
            off += walkers

        # Fit the point cloud to the canvas (de Jong stays within ≈[-2, 2]²).
        minx, maxx = float(px.min()), float(px.max())
        miny, maxy = float(py.min()), float(py.max())
        spanx = max(maxx - minx, 1e-6)
        spany = max(maxy - miny, 1e-6)
        span = max(spanx, spany)
        ccx = (minx + maxx) * 0.5
        ccy = (miny + maxy) * 0.5
        scale = (min(w, h) * 0.9) / span

        fx = (px - ccx) * scale + (w * 0.5)
        fy = (py - ccy) * scale + (h * 0.5)
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix = ix[inside]
        iy = iy[inside]
        vxr = pvx[inside]
        vyr = pvy[inside]

        if ix.size == 0:
            raise RuntimeError("no attractor points landed on the canvas")

        # ── Accumulate density + per-pixel velocity vectors ──
        density = np.zeros((h, w), dtype=np.float64)
        sumvx = np.zeros((h, w), dtype=np.float64)
        sumvy = np.zeros((h, w), dtype=np.float64)
        np.add.at(density, (iy, ix), 1.0)
        np.add.at(sumvx, (iy, ix), vxr.astype(np.float64))
        np.add.at(sumvy, (iy, ix), vyr.astype(np.float64))

        # ── Tone-map density (1 - exp compression → glowing cloud) ──
        dmax = float(density.max())
        # Robust normaliser: 99th percentile of occupied pixels avoids a single
        # hot pixel flattening the whole image.
        occ = density[density > 0]
        p99 = float(np.percentile(occ, 99)) if occ.size else 1.0
        glow = 1.0 - np.exp(-exposure * density / (p99 + 1e-9))
        glow = np.clip(glow, 0.0, 1.0)

        # ── Coloring ──
        if color_mode == "mono":
            color = np.stack([glow, glow, glow], axis=-1)
        elif color_mode == "fire":
            color = np.stack([
                np.clip(glow * 1.5, 0.0, 1.0),
                np.clip(glow * 0.6, 0.0, 1.0),
                np.clip(glow * 0.2, 0.0, 1.0),
            ], axis=-1)
        elif color_mode == "ice":
            color = np.stack([
                np.clip(glow * 0.3, 0.0, 1.0),
                np.clip(0.35 + glow * 0.5, 0.0, 1.0),
                np.clip(0.55 + glow * 0.45, 0.0, 1.0),
            ], axis=-1)
        elif color_mode == "velocity":
            ang = np.arctan2(sumvy, sumvx)            # (h, w)
            hue = (ang / (2.0 * math.pi)) % 1.0      # (h, w)
            color = _iq_ramp(hue)                    # full-brightness hue field
        else:  # density → inferno
            color = _inferno(glow)

        # ── Composite over background ──
        base = {
            "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "navy": np.array([0.04, 0.06, 0.12], dtype=np.float32),
            "cream": np.array([0.96, 0.94, 0.88], dtype=np.float32),
            "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }.get(background, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        base = base.reshape(1, 1, 3)

        if color_mode == "velocity":
            # hue field carries no brightness; let glow dim sparse regions
            out = base * (1.0 - glow[..., None]) + color * glow[..., None]
        else:
            # color already encodes brightness via glow; only fade to bg at edges
            out = base * (1.0 - glow[..., None]) + color

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # ── Sidecar outputs (Rules 4, 5, 6) ──
        density_norm = (density / (dmax + 1e-9)).astype(np.float32)
        write_field(out_dir, density_norm)
        write_scalars(
            out_dir,
            points=float(total),
            a=float(a), b=float(b), c=float(c), d=float(d),
            mean_density=float(float(density.mean())),
            max_density=float(dmax),
            bbox_w=float(spanx), bbox_h=float(spany),
        )
        # PARTICLES: a capped, deterministic sample of visited points (px, py, vx, vy)
        n_p = ix.size
        if n_p > _MAX_PARTICLES:
            pick = rng.choice(n_p, size=_MAX_PARTICLES, replace=False)
            ix_s, iy_s = ix[pick], iy[pick]
            vx_s, vy_s = vxr[pick], vyr[pick]
        else:
            ix_s, iy_s, vx_s, vy_s = ix, iy, vxr, vyr
        particles = np.stack([
            ix_s.astype(np.float32), iy_s.astype(np.float32),
            vx_s.astype(np.float32), vy_s.astype(np.float32),
        ], axis=-1)
        write_particles(out_dir, particles)

        capture_frame("498", out)
        save(out, mn(498, f"De Jong t={_t:.2f}"), out_dir)
        return out
    except Exception as exc:
        fb = np.zeros((int(H), int(W), 3), dtype=np.float32)
        save(fb, mn(498, "De Jong"), out_dir)
        print(f"[method_498] ERROR: {exc}")
        return fb
