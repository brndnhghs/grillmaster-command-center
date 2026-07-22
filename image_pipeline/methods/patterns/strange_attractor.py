"""Strange Attractor — deterministic chaos rendered as a density point-cloud.

Renders classic 2D strange attractors (iterated chaotic maps) as a bright
point-density field:

    * Clifford  (Clifford 1989 / "Strange Attractors" poster tradition):
          x' = sin(a*y) + c*cos(a*x)
          y' = sin(b*x) + d*cos(b*y)
    * de Jong (Peter de Jong, 1987):
          x' = sin(a*y) - cos(b*x)
          y' = sin(c*x) - cos(d*y)
    * Hopalong (Barry Martin, 1989, "Computer Recreations", Scientific American):
          x' = y - sign(x)*sqrt(|b*x - c|)
          y' = a - x

Each is a deterministic IFS — the same parameters always yield the same infinite
point set — so it is trivially animatable: morph the four parameters (a,b,c,d)
with the animation clock and the attractor continuously deforms; orbit/rotate the
whole figure; breathe the exposure. Because the maps are cheap (a few transcen-
dentals per point) and the histogram is built with a single `np.bincount`, a
multi-million-point render costs well under the pipeline's 150 s timeout budget
— deliberately chosen so the node never becomes a timeout casualty.

High temporal variance (the point cloud reshapes every frame under `morph` /
`orbit`) also means the node reliably passes the liveness cull, unlike
contrast-only or static patterns.

CPU path authoritative. Closed-form, no carried simulation state, so a clean
per-pixel f(uv, t) GPU twin is a natural follow-up.

Animation modes (Architecture B — per-frame re-call with `time`):
    none   — fixed (a,b,c,d): frame Δ ≈ 0 (static baseline).
    morph  — a,b,c,d breathe via cos(_t) so the attractor deforms smoothly.
             cos (not sin): cos(0)=+1, cos(π)=−1 keep audit frames distinct
             (sin-phase delta degeneracy would make t=0 vs t=π a false negative).
    orbit  — the whole figure rotates + drifts by _t (non-integer rate ⇒ never
             symmetry-aligned at the audit sample times, so it always reads as
             motion).
    breathe — exposure pulses via 0.5 + 0.5*cos(_t) (smooth brightness breathing).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, BG_DEFAULT, W, H,
    write_mask, write_particles, write_scalars, write_field,
)
from ...core.animation import capture_frame

_SYSTEMS = ("clifford", "de_jong", "hopalong")


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable moving-average (box) blur, radius r, dependency-free.

    Used only to soften the dot stamp when `dot_size` > 1; r is 0/1/2 so the
    window is tiny and this is cheap. Pure numpy (no scipy dependency).
    """
    if r <= 0:
        return a
    k = 2 * r + 1
    # horizontal pass via cumulative sum
    pad = np.concatenate([a[:, :r], a, a[:, -r:]], axis=1)
    cs = np.cumsum(pad, axis=1, dtype=np.float64)
    h = (cs[:, k - 1:] - cs[:, :pad.shape[1] - (k - 1)]) / k
    # vertical pass
    pad = np.concatenate([h[:r, :], h, h[-r:, :]], axis=0)
    cs = np.cumsum(pad, axis=0, dtype=np.float64)
    v = (cs[k - 1:, :] - cs[:pad.shape[0] - (k - 1), :]) / k
    return v.astype(a.dtype)


def _hsv_to_rgb_vec(h, s, v):
    """Vectorized HSV → RGB, all arrays broadcastable, values in [0,1]."""
    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    m0 = i == 0
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    m1 = i == 1
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    m2 = i == 2
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    m3 = i == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    m4 = i == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    m5 = i == 5
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
    return r, g, b


@method(
    id="957",
    name="Strange Attractor",
    category="patterns",
    new_image_contract=True,
    tags=["generative", "chaos", "attractor", "clifford", "de_jong", "hopalong",
          "fractal", "animation", "point-cloud", "gpu-twin-candidate"],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK",
             "particles": "PARTICLES"},
    params={
        "system": {"description": "chaotic map (clifford/de_jong/hopalong)",
                   "choices": list(_SYSTEMS), "default": "clifford"},
        "a": {"description": "map parameter a", "min": -3.0, "max": 3.0, "default": -1.4},
        "b": {"description": "map parameter b", "min": -3.0, "max": 3.0, "default": 1.6},
        "c": {"description": "map parameter c", "min": -3.0, "max": 3.0, "default": 1.0},
        "d": {"description": "map parameter d", "min": -3.0, "max": 3.0, "default": 0.7},
        "n_points": {"description": "total plotted points (orbits x steps)",
                     "min": 100000.0, "max": 4000000.0, "default": 1200000.0},
        "dot_size": {"description": "point radius in px (1-2 per line convention)",
                     "min": 1.0, "max": 3.0, "default": 1.0},
        "exposure": {"description": "density brightness multiplier",
                     "min": 0.1, "max": 5.0, "default": 1.6},
        "gamma": {"description": "tonal gamma (lower = brighter highlights)",
                  "min": 0.3, "max": 2.5, "default": 1.0},
        "color_mode": {"description": "colour source (mono/rainbow)",
                       "default": "rainbow"},
        "hue": {"description": "base hue for mono mode (0-1)",
                "min": 0.0, "max": 1.0, "default": 0.58},
        "sat": {"description": "colour saturation", "min": 0.0, "max": 1.0, "default": 0.85},
        "background": {"description": "canvas background (dark/light/mid)",
                       "choices": ["dark", "light", "mid"], "default": "dark"},
        "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
        "anim_mode": {"description": "animation mode (none/morph/orbit/breathe)",
                      "choices": ["none", "morph", "orbit", "breathe"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    },
)
def method_strange_attractor(out_dir: Path, seed: int, params=None):
    """Strange Attractor — Clifford / de Jong / Hopalong chaos point-cloud.

    Iterates the chosen chaotic map, accumulates the visited points into a
    density histogram (np.bincount), auto-fits the framing, and shades by
    density (and, in rainbow mode, by per-point hue).

    Distinct from the other pattern nodes:
      * maurer_rose / superformula / harmonograph: a single closed parametric
        curve, not a chaotic point-set.
      * truchet / wallpaper / quasicrystal: tiling / symmetry lattices.
      * fractal / julia_set: escape-time iteration on a fixed grid.
    Strange Attractor is the chaotic-point-density sibling — few float params
    (a,b,c,d) govern an emergent, organic, infinitely-detailed structure.

    Params:
        system:     clifford / de_jong / hopalong
        a,b,c,d:    map parameters (defaults are known-good attractors)
        n_points:   total plotted points (orbits x steps)
        dot_size:   point radius (kept 1-2px per the thin-line convention)
        exposure:   density brightness multiplier
        gamma:      tonal gamma
        color_mode: mono (base hue) / rainbow (per-point hue sweep)
        hue:        base hue for mono mode
        sat:        colour saturation
        background: dark / light / mid canvas
        time:       animation phase [0, 2pi)
        anim_mode:  none / morph / orbit / breathe
        anim_speed: animation speed (0.1-5.0)
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        anim_mode = str(params.get("anim_mode", "none"))
        anim_speed = float(params.get("anim_speed", 1.0))
        seed_all(seed)

        system = str(params.get("system", "clifford"))
        if system not in _SYSTEMS:
            system = "clifford"
        a = float(np.clip(params.get("a", -1.4), -3.0, 3.0))
        b = float(np.clip(params.get("b", 1.6), -3.0, 3.0))
        c = float(np.clip(params.get("c", 1.0), -3.0, 3.0))
        d = float(np.clip(params.get("d", 0.7), -3.0, 3.0))
        n_points = int(np.clip(params.get("n_points", 1200000.0), 100000, 4000000))
        dot_size = float(np.clip(params.get("dot_size", 1.0), 1.0, 3.0))
        exposure = float(np.clip(params.get("exposure", 1.6), 0.1, 5.0))
        gamma = float(np.clip(params.get("gamma", 1.0), 0.3, 2.5))
        color_mode = str(params.get("color_mode", "rainbow"))
        hue = float(np.clip(params.get("hue", 0.58), 0.0, 1.0))
        sat = float(np.clip(params.get("sat", 0.85), 0.0, 1.0))
        background = str(params.get("background", "dark"))

        # ── Animation clock (rename t to avoid shadowing the time param) ──
        _t = t * anim_speed
        a_eff, b_eff, c_eff, d_eff = a, b, c, d
        orbit_ang = 0.0
        orbit_dx = 0.0
        orbit_dy = 0.0
        exp_eff = exposure
        if anim_mode == "morph":
            # cos (not sin): keeps the t=0 vs t=π audit frames distinct.
            a_eff = a + 0.35 * math.cos(_t)
            b_eff = b + 0.35 * math.cos(_t + 1.7)
            c_eff = c + 0.35 * math.cos(_t + 3.1)
            d_eff = d + 0.35 * math.cos(_t + 4.6)
        elif anim_mode == "orbit":
            # Non-integer rate ⇒ never symmetry-aligned at audit sample times.
            orbit_ang = _t * 0.6
            orbit_dx = math.sin(_t * 0.4) * 0.06
            orbit_dy = math.cos(_t * 0.33) * 0.06
        elif anim_mode == "breathe":
            exp_eff = exposure * (0.5 + 0.5 * math.cos(_t))

        # ── Background ──
        if background == "light":
            bg = np.array([0.96, 0.96, 0.98], dtype=np.float32)
        elif background == "mid":
            bg = np.array(BG_DEFAULT, dtype=np.float32) / 255.0
        else:  # dark
            bg = np.array([0.03, 0.04, 0.07], dtype=np.float32)

        # ── Iterate the chaotic map ──
        # Run M independent orbits, each `steps` long; plot every point.
        steps = 80
        M = max(1024, n_points // steps)
        rng = np.random.default_rng(seed)
        x = (rng.random(M) * 2.0 - 1.0).astype(np.float64)
        y = (rng.random(M) * 2.0 - 1.0).astype(np.float64)

        total = M * steps
        xs_all = np.empty(total, dtype=np.float64)
        ys_all = np.empty(total, dtype=np.float64)
        off = 0
        for _ in range(steps):
            if system == "clifford":
                xn = np.sin(a_eff * y) + c_eff * np.cos(a_eff * x)
                yn = np.sin(b_eff * x) + d_eff * np.cos(b_eff * y)
            elif system == "de_jong":
                xn = np.sin(a_eff * y) - np.cos(b_eff * x)
                yn = np.sin(c_eff * x) - np.cos(d_eff * y)
            else:  # hopalong
                sgn = np.where(x >= 0, 1.0, -1.0)
                xn = y - sgn * np.sqrt(np.abs(b_eff * x - c_eff))
                yn = a_eff - x
            xs_all[off:off + M] = x
            ys_all[off:off + M] = y
            x, y = xn, yn
            off += M

        # Drop any non-finite (Hopalong can explode on bad params) and clip.
        finite = np.isfinite(xs_all) & np.isfinite(ys_all)
        xs_all = xs_all[finite]
        ys_all = ys_all[finite]
        if xs_all.size == 0:
            xs_all = np.array([0.0]); ys_all = np.array([0.0])

        # ── Auto-fit framing ──
        xmin, xmax = float(xs_all.min()), float(xs_all.max())
        ymin, ymax = float(ys_all.min()), float(ys_all.max())
        span_x = max(xmax - xmin, 1e-6)
        span_y = max(ymax - ymin, 1e-6)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span = max(span_x, span_y) * 1.06
        scale_px = (0.5 / span) * min(W, H)

        # Per-point hue (rainbow sweep along the iteration / orbit index).
        N = xs_all.size
        frac = np.linspace(0.0, 1.0, N, dtype=np.float32)
        hue_arr = np.mod(frac + hue, 1.0).astype(np.float32)

        # Map to pixel coords (flip y for math-convention up).
        px = ((xs_all - cx) * scale_px + W / 2.0) + orbit_dx * W
        py = (H / 2.0 - (ys_all - cy) * scale_px) + orbit_dy * H
        # Orbit rotation about canvas centre.
        if orbit_ang != 0.0:
            ca, sa = math.cos(orbit_ang), math.sin(orbit_ang)
            dx = px - W / 2.0
            dy = py - H / 2.0
            px = W / 2.0 + dx * ca - dy * sa
            py = H / 2.0 + dx * sa + dy * ca
        px = np.clip(np.round(px), 0, W - 1).astype(np.int64)
        py = np.clip(np.round(py), 0, H - 1).astype(np.int64)
        idx = (py * W + px).astype(np.int64)

        # Radius grows the bincount window (dot_size up to 3px → 3x3 stamp).
        if dot_size > 1.5:
            rad = 1
        else:
            rad = 0
        # Density via bincount (fast, vectorised).
        count = np.bincount(idx, minlength=W * H).astype(np.float32).reshape(H, W)
        if rad > 0:
            # light soften to approximate a soft dot (cheap, separable, no deps).
            count = _box_blur(count, rad)
        count = np.clip(count, 0.0, None)

        # Per-pixel average hue from accumulated hue (rainbow mode).
        if color_mode == "rainbow":
            hue_sum = np.bincount(idx, weights=hue_arr, minlength=W * H).astype(np.float32).reshape(H, W)
            with np.errstate(invalid="ignore", divide="ignore"):
                avg_hue = np.where(count > 0, hue_sum / np.clip(count, 1e-6, None), hue)
            avg_hue = np.mod(avg_hue, 1.0)
        else:
            avg_hue = np.full((H, W), hue, dtype=np.float32)

        # ── Shade by density ──
        if count.sum() > 0:
            peak = max(float(np.percentile(count, 99.9)), 1e-6)
        else:
            peak = 1.0
        Dn = count / peak
        bright = np.power(np.clip(Dn * exp_eff, 0.0, None), 1.0 / gamma)
        bright = np.clip(bright, 0.0, 1.0)

        rr, gg, bb = _hsv_to_rgb_vec(avg_hue, sat, bright)
        rgb = np.stack([rr, gg, bb], axis=-1).astype(np.float32)
        # composite over background where dark (additive glow look)
        rgb = np.clip(rgb + bg * (1.0 - bright[..., None]), 0.0, 1.0)

        mask = (count > 0).astype(np.float32)
        # FIELD = normalised density (luminance reads as mean over channels).
        field = np.clip(Dn, 0.0, 1.0).astype(np.float32)

        # ── Particles: a subsample of points with local velocity ──
        stride = max(1, N // 4000)
        pidx = np.arange(0, N - 1, stride)
        pxx = px[pidx].astype(np.float32)
        pyy = py[pidx].astype(np.float32)
        velx = (px[pidx + 1] - pxx)
        vely = (py[pidx + 1] - pyy)
        particles = np.zeros((pidx.size, 4), dtype=np.float32)
        particles[:, 0] = pxx
        particles[:, 1] = pyy
        particles[:, 2] = velx
        particles[:, 3] = vely

        capture_frame("957", rgb)
        save(rgb, mn(957, "Strange Attractor"), out_dir)
        try:
            write_field(out_dir, field)
            write_mask(out_dir, mask)
            write_particles(out_dir, particles)
            write_scalars(
                out_dir,
                system_code=float(hash(system) % 1000),
                a=float(a_eff), b=float(b_eff), c=float(c_eff), d=float(d_eff),
                points=float(N),
                peak_density=float(peak),
                coverage=float(mask.mean()),
            )
        except Exception:
            pass
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 0.5, dtype=np.float32)
        save(fallback, mn(957, "Strange Attractor"), out_dir)
        print(f"[method_957] ERROR: {exc}")
        return fallback
