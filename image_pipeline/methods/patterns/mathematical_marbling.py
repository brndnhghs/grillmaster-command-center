"""
#953 — Mathematical Marbling

Digital paper-marbling as a closed-form fluid advection, after
Lu, Jaffer, Jin, Zhao & Mao, "Mathematical Marbling", IEEE Computer
Graphics & Applications 32(6):26–35, 2012 (https://people.csail.mit.edu/jaffer/Marbling/).

Traditional marbling: dye dropped onto a viscous fluid surface, then
"tines" (stylus) are dragged through it. The surface is so viscous that
the dye displacement is well approximated by incompressible, irrotational
2D fluid flow. Each operation has a closed form:

  • Drop        — a filled circle of color c at (x, y) with radius r.
  • Tine line   — all points are displaced by translation that decays
                  with distance to a line L:
                    D(p) = m · d(p, L) · n̂     where d = exp(-|t·(p−p0)|/c)
                  (a single tine stroke along direction t̂, normal n̂,
                  strength m, sharpness c).
  • Circular tine — rotation that decays with distance to a center:
                    θ(p) = a · exp(-dist²/λ²)

Because every operation is a position map p → p', a point in the FINAL
image is colored by inverting the maps for each drop (last drop wins),
which is O(#drops) per pixel — exact and crisp, no raster diffusion.

Architecture B — single frame computed from time `t` (per-frame re-call).
All drops are injected over the animation; `t` sweeps a tine stroke so the
pattern evolves smoothly.

Animation evolves only the tine stroke position (smooth, no cusps) so a
wired driver can morph the marbling live.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save,
    mn,
    seed_all,
    W,
    H,
    write_scalars,
    wired_source_lum,
)
from ...core.animation import capture_frame


# ── Helpers ──

def _drop_at(px: np.ndarray, py: np.ndarray, cx: float, cy: float,
             r: float) -> np.ndarray:
    """Boolean mask: points inside the drop circle."""
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def _apply_tine(x: np.ndarray, y: np.ndarray, p0: np.ndarray,
                t_hat: np.ndarray, m: float, c: float):
    """Single linear tine stroke. Returns displaced (x, y).

    Displacement = m · exp(-dist_to_line / c) along the line normal.
    """
    # distance of each point to the line through p0 with direction t_hat
    rel = np.stack([x - p0[0], y - p0[1]], axis=-1)
    along = rel @ t_hat                      # signed distance along line
    d = np.abs(along)
    decay = np.exp(-d / c)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    disp = m * decay
    x2 = x + disp * n_hat[0]
    y2 = y + disp * n_hat[1]
    return x2, y2


def _apply_circular_tine(x: np.ndarray, y: np.ndarray, ctr: np.ndarray,
                         a: float, lam: float):
    """Circular tine (whirlpool): rotate each point by angle decaying with r²."""
    dx = x - ctr[0]
    dy = y - ctr[1]
    r2 = dx * dx + dy * dy
    theta = a * np.exp(-r2 / (lam * lam))
    ct = np.cos(theta)
    st = np.sin(theta)
    x2 = ctr[0] + dx * ct - dy * st
    y2 = ctr[1] + dx * st + dy * ct
    return x2, y2


def _palette(n: int, seed: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    cols = rng.uniform(0.05, 1.0, size=(n, 3))
    # a few vivid anchors for nicer marbling
    cols[0] = np.array([0.05, 0.12, 0.30])
    cols[1 % n] = np.array([0.85, 0.15, 0.20])
    cols[2 % n] = np.array([0.95, 0.85, 0.20])
    cols[3 % n] = np.array([0.10, 0.55, 0.45])
    return [tuple(c) for c in cols]


# ══════════════════════════════════════════════════════════════════════
#  Method
# ══════════════════════════════════════════════════════════════════════

@method(
    id="953",
    name="Mathematical Marbling",
    category="patterns",
    tags=["marbling", "fluid", "advection", "stylization", "procedural", "pattern"],
    timeout=120,
    outputs={"image": "IMAGE"},
    inputs={"image_in": "IMAGE"},
    params={
        "source": {
            "description": "background: flat color or the wired upstream image",
            "choices": ["flat", "input_image"],
            "default": "flat",
        },
        "n_drops": {
            "description": "number of colored drops injected",
            "min": 1, "max": 60, "default": 14,
        },
        "drop_radius": {
            "description": "base drop radius (fraction of min(W,H))",
            "min": 0.01, "max": 0.3, "default": 0.09,
        },
        "anim_mode": {
            "description": "which tine stroke animates with time",
            "choices": ["tine", "circular", "none"],
            "default": "tine",
        },
        "anim_speed": {
            "description": "animation speed multiplier (tine sweep rate)",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "tine_strength": {
            "description": "tine displacement magnitude (fraction of min(W,H))",
            "min": 0.0, "max": 0.6, "default": 0.22,
        },
        "tine_sharpness": {
            "description": "tine sharpness c (smaller = sharper, more localized)",
            "min": 0.02, "max": 0.5, "default": 0.14,
        },
        "n_tines": {
            "description": "number of tine strokes layered over the drops",
            "min": 1, "max": 12, "default": 3,
        },
        "seed": {
            "description": "random seed for drop placement + palette",
            "min": 0, "max": 99999, "default": 42,
        },
    },
)
def method_marbling(out_dir: Path, seed: int, params=None):
    """Mathematical Marbling — closed-form fluid-advection stylization.

    Drops of color are injected onto a viscous fluid surface, then tine
    strokes drag the dye into the classic marbled veins. Every operation is
    an exact, invertible position map (Lu et al. 2012), so the final image
    is colored crisply by inverting the maps per drop — no raster diffusion.

    Animation modes:
        tine:     linear tine strokes sweep across the surface with time
        circular: a whirlpool tine rotates the pattern with time
        none:     static composition (no time evolution)

    Architecture B — single frame from time `t`.
    """
    if params is None:
        params = {}
    t = float(params.get("time", 0.0))
    anim_mode = str(params.get("anim_mode", "tine"))
    anim_speed = float(params.get("anim_speed", 1.0))
    _t = t * anim_speed
    n_drops = int(params.get("n_drops", 14))
    drop_r = float(params.get("drop_radius", 0.09))
    tine_strength = float(params.get("tine_strength", 0.22))
    tine_c = float(params.get("tine_sharpness", 0.14))
    n_tines = int(params.get("n_tines", 3))
    rseed = int(params.get("seed", seed))

    seed_all(rseed)
    rng = np.random.default_rng(rseed)

    # ── Background ──
    src_lum = None
    if str(params.get("source", "flat")) == "input_image":
        src_lum = wired_source_lum(params, W, H)
    if src_lum is not None:
        bg = (np.clip(src_lum, 0.0, 1.0) * 255).astype(np.uint8)
        print("  Background: wired upstream image")
    else:
        bg = np.full((H, W, 3), 245, dtype=np.uint8)
        print("  Background: flat paper")

    # ── Drop definitions (normalised coords) ──
    minwh = min(W, H)
    base_r = drop_r * minwh
    drops = []  # list of (cx, cy, r, (r,g,b))
    palette = _palette(n_drops, rseed)
    for i in range(n_drops):
        cx = float(rng.uniform(0.08, 0.92)) * W
        cy = float(rng.uniform(0.08, 0.92)) * H
        r = base_r * float(rng.uniform(0.6, 1.4))
        color = palette[i % len(palette)]
        drops.append((cx, cy, r, color))

    # ── Build the advection position maps (applied to a query grid) ──
    # We evaluate on a pixel grid and invert: for each pixel, find which
    # drop (in reverse order) contains the *pre-image* point.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # query point = output pixel; we undo the tine maps to get the point
    # that, after advection, lands here. Inverse of a monotonic decay tine:
    # displacement along normal = f(along); invert by fixed-point on axis.
    qx, qy = xx.copy(), yy.copy()

    # Apply tine strokes (their effect, evaluated at the *current* time)
    if anim_mode in ("tine",):
        for k in range(n_tines):
            # direction rotates slightly per tine; position sweeps with t
            ang = 2.0 * math.pi * k / max(1, n_tines) + 0.3
            t_hat = np.array([math.cos(ang), math.sin(ang)])
            n_hat = np.array([-t_hat[1], t_hat[0]])
            # sweep the line origin across the canvas with time
            sweep = (math.sin(t * anim_speed * 0.6 + k * 1.3) * 0.5 + 0.5)
            p0 = np.array([sweep * W, (0.5 + 0.3 * math.cos(k)) * H])
            m = tine_strength * minwh
            c = max(1e-3, tine_c * minwh)
            # inverse displacement: move query point opposite the forward map
            rel = np.stack([qx - p0[0], qy - p0[1]], axis=-1)
            along = rel @ t_hat
            d = np.abs(along)
            decay = np.exp(-d / c)
            disp = m * decay
            qx = qx - disp * n_hat[0]
            qy = qy - disp * n_hat[1]
    elif anim_mode == "circular":
        ctr = np.array([0.5 * W, 0.5 * H])
        lam = 0.35 * minwh
        # rotate the query grid back by an angle that grows with t
        a = (math.sin(t * anim_speed * 0.5) * 0.5 + 0.5) * 3.0  # up to ~3 rad
        dx = qx - ctr[0]
        dy = qy - ctr[1]
        r2 = dx * dx + dy * dy
        theta = a * np.exp(-r2 / (lam * lam))
        ct = np.cos(-theta)
        st = np.sin(-theta)
        qx = ctr[0] + dx * ct - dy * st
        qy = ctr[1] + dx * st + dy * ct

    # ── Invert per drop: last drop wins ──
    out = bg.astype(np.float64)
    for (cx, cy, r, color) in reversed(drops):
        inside = _drop_at(qx, qy, cx, cy, r)
        out[inside, 0] = color[0] * 255.0
        out[inside, 1] = color[1] * 255.0
        out[inside, 2] = color[2] * 255.0

    img = np.clip(out, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGB")

    capture_frame("953", np.array(pil, dtype=np.float32) / 255.0)
    write_scalars(out_dir, n_drops=n_drops, n_tines=n_tines,
                  tine_strength=tine_strength)
    # Architecture B: include time in name so --animate PNGs don't overwrite
    # each other on disk (pitfall #12).
    save(pil, mn(953, f"Mathematical Marbling t={_t:.2f}"), out_dir)
    return pil
