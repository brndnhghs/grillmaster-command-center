from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field, PALETTES, BG_DEFAULT,
)
from ...core.animation import capture_frame


# ── hsv -> rgb (vectorized, all in [0,1]) ──
def _hsv2rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = h - np.floor(h)
    i = np.floor(h * 6.0).astype(np.int64)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    for k in range(6):
        m = i % 6 == k
        if k == 0:
            r[m], g[m], b[m] = v[m], t[m], p[m]
        elif k == 1:
            r[m], g[m], b[m] = q[m], v[m], p[m]
        elif k == 2:
            r[m], g[m], b[m] = p[m], v[m], t[m]
        elif k == 3:
            r[m], g[m], b[m] = p[m], q[m], v[m]
        elif k == 4:
            r[m], g[m], b[m] = t[m], p[m], v[m]
        else:
            r[m], g[m], b[m] = v[m], p[m], q[m]
    return np.stack([r, g, b], axis=-1)


def _palette_rgb(pal_name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n RGB triples in [0,1] from a named PALETTES entry."""
    pal = PALETTES.get(pal_name, PALETTES.get("vapor", [(255, 255, 255)]))
    arr = np.asarray(pal, dtype=np.float64) / 255.0
    idx = rng.integers(0, len(arr), size=n)
    return arr[idx]


@method(id="342", name="Strange Attractor 2D", category="patterns",
        tags=["chaos", "strange-attractor", "clifford", "de-jong", "generative", "animation"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD", "density": "FIELD"},
        params={
    "attractor": {"description": "map family: clifford (Peter de Jong's original) or dejong (De Jong's own form)", "default": "clifford"},
    "a": {"description": "parameter a (shape of the attractor)", "min": -3.0, "max": 3.0, "default": -1.4},
    "b": {"description": "parameter b", "min": -3.0, "max": 3.0, "default": 1.6},
    "c": {"description": "parameter c", "min": -3.0, "max": 3.0, "default": 1.0},
    "d": {"description": "parameter d", "min": -3.0, "max": 3.0, "default": 0.7},
    "points": {"description": "trajectories integrated (density resolution)", "min": 200_000, "max": 8_000_000, "default": 2_000_000},
    "color_by": {"description": "how to color points: density, speed (local velocity), or palette (random per point)", "default": "density"},
    "palette": {"description": "palette used when color_by=palette", "default": "vapor"},
    "brightness": {"description": "overall brightness multiplier", "min": 0.2, "max": 3.0, "default": 1.2},
    "bg_style": {"description": "background (dark/light)", "default": "dark"},
    "anim_mode": {"description": "animation mode: none, mutate (params sweep), or rotate (rotate the plane)", "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_strange_attractor_2d(out_dir, seed: int, params=None):
    """Render a 2D strange attractor (Clifford / De Jong) as a density field.

    Technique (Peter de Jong, "The Universe of Discourse", 1987; the Clifford
    attractor is the same recurrence with an extra sin on the second component):
    a discrete-time map that, for almost all parameter choices, settles onto a
    fractal dust of points. The trajectory is NOT random — it is fully
    deterministic, but highly sensitive to (a,b,c,d). We integrate millions of
    steps, accumulate a 2D histogram (the attractor's invariant density), then
    tone-map it. This is the classic "strange attractor" generative-art look.

    Two forms:
      * clifford:  x' = sin(a*y) + c*cos(a*x) ;  y' = sin(b*x) + d*cos(b*y)
      * dejong:    x' = sin(a*y) - cos(b*x)   ;  y' = sin(c*x) - cos(d*y)

    Coloring options:
      * density : logarithmic tone-map of the point density (the canonical look)
      * speed   : local |step| velocity -> hue rainbow (reveals the map's flow)
      * palette : each point gets a random palette color, blended in density

    Architecture B (closed-form per frame): the orchestrator re-calls with an
    increasing ``time``. Animation modes are smooth and seed-stable:
      * mutate  : (a,b,c,d) drift along Lissajous paths -> endless morphing shapes
      * rotate  : the attractor plane slowly rotates about the canvas center
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        attractor = params.get("attractor", "clifford")
        a = float(params.get("a", -1.4))
        b = float(params.get("b", 1.6))
        c = float(params.get("c", 1.0))
        d = float(params.get("d", 0.7))
        points = int(params.get("points", 2_000_000))
        color_by = params.get("color_by", "density")
        pal_name = params.get("palette", "vapor")
        brightness = float(params.get("brightness", 1.2))
        bg_style = params.get("bg_style", "dark")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation time wiring (Architecture B) ──
        _t = t * anim_speed if anim_mode != "none" else 0.0

        if anim_mode == "mutate":
            # Smoothly interpolate between two KNOWN-GOOD attractor parameter
            # sets. Unbounded drift would often push (a,b,c,d) into a degenerate
            # region (fixed point / near-empty set); interpolating between two
            # validated presets keeps every frame visually rich AND animates.
            # C-infinity weight -> no cusps.
            w = 0.5 + 0.5 * math.sin(_t * 0.5)
            a0, b0, c0, d0 = a, b, c, d
            a1, b1, c1, d1 = 1.7, 1.7, 0.6, 1.2
            a = a0 * (1.0 - w) + a1 * w
            b = b0 * (1.0 - w) + b1 * w
            c = c0 * (1.0 - w) + c1 * w
            d = d0 * (1.0 - w) + d1 * w
        rotate = _t if anim_mode == "rotate" else 0.0

        # ── Integrate the recurrence (vectorized on chunks) ──
        # The attractor lives roughly in [-2, 2]^2 (clifford) / [-2, 2] (dejong).
        n = max(1, min(points, 8_000_000))
        BINS = 900  # density grid resolution (downsample to canvas below)
        span = 2.2
        # random initial condition (one seed point is enough; the orbit covers the set)
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)

        # We accumulate point positions + per-point speed for coloring.
        px = np.empty(n, dtype=np.float32)
        py = np.empty(n, dtype=np.float32)
        pspeed = np.empty(n, dtype=np.float32)
        CH = 200_000
        # warm-up to land on the attractor (skip transient)
        for _ in range(200):
            if attractor == "dejong":
                nx = math.sin(a * y) - math.cos(b * x)
                ny = math.sin(c * x) - math.cos(d * y)
            else:
                nx = math.sin(a * y) + c * math.cos(a * x)
                ny = math.sin(b * x) + d * math.cos(b * y)
            x, y = nx, ny

        for start in range(0, n, CH):
            end = min(start + CH, n)
            chunk = end - start
            xs = np.empty(chunk, dtype=np.float32)
            ys = np.empty(chunk, dtype=np.float32)
            sp = np.empty(chunk, dtype=np.float32)
            cxv, cyv = x, y
            for k in range(chunk):
                if attractor == "dejong":
                    nx = math.sin(a * cyv) - math.cos(b * cxv)
                    ny = math.sin(c * cxv) - math.cos(d * cyv)
                else:
                    nx = math.sin(a * cyv) + c * math.cos(a * cxv)
                    ny = math.sin(b * cxv) + d * math.cos(b * cyv)
                dxs = nx - cxv
                dys = ny - cyv
                sp[k] = math.sqrt(dxs * dxs + dys * dys)
                xs[k] = nx
                ys[k] = ny
                cxv, cyv = nx, ny
            x, y = cxv, cyv
            px[start:end] = xs
            py[start:end] = ys
            pspeed[start:end] = sp

        # ── Optional plane rotation (animation only) ──
        if rotate != 0.0:
            ca, sa = math.cos(rotate), math.sin(rotate)
            rx = px * ca - py * sa
            ry = px * sa + py * ca
            px, py = rx, ry

        # ── Accumulate density into the BINS grid ──
        gx = np.clip(((px + span) / (2 * span) * BINS).astype(np.int64), 0, BINS - 1)
        gy = np.clip(((py + span) / (2 * span) * BINS).astype(np.int64), 0, BINS - 1)
        density = np.zeros((BINS, BINS), dtype=np.float64)
        np.add.at(density, (gy, gx), 1.0)

        # ── Build the RGB image at canvas resolution via downsample + tone-map ──
        # resize density to (H, W) by block averaging
        fy = BINS / int(H)
        fx = BINS / int(W)
        ys_idx = (np.arange(int(H)) * fy).clip(0, BINS - 1).astype(np.int64)
        xs_idx = (np.arange(int(W)) * fx).clip(0, BINS - 1).astype(np.int64)
        dens_small = density[np.ix_(ys_idx, xs_idx)]

        # log tone-map (canonical for attractor density)
        eps = 1.0
        ld = np.log(dens_small + eps)
        ld_min, ld_max = ld.min(), ld.max()
        ld_n = (ld - ld_min) / max(ld_max - ld_min, 1e-9)

        if bg_style == "light":
            bg = np.full((H, W, 3), 0.92, dtype=np.float64)
        else:
            bg = np.zeros((H, W, 3), dtype=np.float64)

        if color_by == "speed":
            # per-pixel mean step-speed -> hue rainbow (reveals the map's flow)
            sp_small = np.zeros((BINS, BINS), dtype=np.float64)
            np.add.at(sp_small, (gy, gx), pspeed)
            sp_small = sp_small[np.ix_(ys_idx, xs_idx)]
            spn = sp_small / max(sp_small.max(), 1e-9)
            hue = spn
            sat = np.clip(0.4 + ld_n, 0.0, 1.0)
            val = np.clip(ld_n * brightness, 0.0, 1.0)
            col = _hsv2rgb(hue, sat, val)
            mask = ld_n > 0.0
            rgb = bg.copy()
            rgb[mask] = col[mask]
        elif color_by == "palette":
            # deterministic per-pixel hue from a hash of the bin, tints density
            hue = ((gx.astype(np.float64) * 0.013 + gy.astype(np.float64) * 0.029 + 0.3) % 1.0)[np.ix_(ys_idx, xs_idx)]
            sat = np.clip(0.3 + ld_n * 0.7, 0.0, 1.0)
            val = np.clip((0.15 + ld_n) * brightness, 0.0, 1.0)
            col = _hsv2rgb(hue, sat, val)
            mask = ld_n > 0.0
            rgb = bg.copy()
            rgb[mask] = col[mask]
        else:  # density
            val = np.clip(ld_n * brightness, 0.0, 1.0)
            if bg_style == "light":
                # dark ink on light paper
                rgb = bg * (1.0 - val) + np.stack([val * 0.1, val * 0.12, val * 0.2], -1)
            else:
                rgb = np.stack([val, val, val], -1)

        rgb = rgb.astype(np.float32)

        # ── Provenance + fields (Rule 4 / Rule 5) ──
        write_scalars(out_dir,
                      params_a=float(a), params_b=float(b), params_c=float(c), params_d=float(d),
                      occupied_fraction=float(float((density > 0).sum()) / density.size),
                      peak_density=float(float(density.max())))
        write_field(out_dir, ld_n.astype(np.float32))

        capture_frame("342", rgb)
        save(rgb, mn(342, f"StrangeAttractor2D t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), BG_DEFAULT[0], dtype=np.uint8)
        save(fallback, mn(342, "StrangeAttractor2D"), out_dir)
        print(f"[method_342] ERROR: {exc}")
        return fallback
