from __future__ import annotations

import math

import numpy as np

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, get_canvas, write_scalars, write_field, write_mask,
)
from ...core.animation import capture_frame


# ── Metaballs (implicit blobby surfaces) ──────────────────────────────────────
# A classic real-time *implicit surface* technique: each ball contributes a
# radial potential F_i(p) = r_i^2 / |p - c_i|^2, and the total scalar field is
# F(p) = Σ F_i(p). A pixel is "inside" the goo when F(p) ≥ threshold (≈1). The
# smooth falloff gives the characteristic fused, liquid blobs that separate as
# balls move apart. We render the field with a soft (smoothstep) edge and color
# each pixel by the density-weighted blend of the balls' hues, so neighbouring
# blobs merge with a gooey gradient. The animation is *structural* — ball
# positions (orbit) and radii (pulse) actually move — so it passes any temporal
# liveness test without an external driver, unlike a contrast-only flicker.
#
# This is a closed-form Architecture-B method: the orchestrator re-calls it per
# frame with increasing `time`; anim_mode="none" is a pure function of
# (seed, params) -> static baseline (Δ ≈ 0).


def _hsv_to_rgb(h, s, v):
    i = int(math.floor(h * 6.0)) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = [v, q, p, p, t, v][i]
    g = [t, v, v, q, p, p][i]
    b = [p, p, t, v, v, q][i]
    return r, g, b


def _inferno(t: np.ndarray) -> np.ndarray:
    c0 = np.array([0.00021894, 0.00165100, -0.01948090])
    c1 = np.array([0.10651342, 0.56395644, 3.93271239])
    c2 = np.array([11.60249308, -3.97285397, -15.94239411])
    c3 = np.array([-41.70399613, 17.43639888, 44.35414520])
    c4 = np.array([77.16293570, -33.40235894, -81.80730926])
    c5 = np.array([-71.31942824, 32.62606426, 73.20951986])
    c6 = np.array([25.13112622, -12.24266895, -23.07032500])
    out = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    return np.clip(out, 0.0, 1.0)


def _smoothstep(e0, e1, x):
    tt = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return tt * tt * (3.0 - 2.0 * tt)


@method(id="505", name="Metaballs", category="patterns",
        tags=["metaballs", "implicit-surface", "blobby", "procedural",
              "animation", "scalar-field", "real-time-cg"],
        inputs={},
        outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
        params={
    "balls": {"description": "number of metaballs (blobs)",
              "min": 2, "max": 16, "default": 8},
    "ball_size": {"description": "base ball radius as fraction of canvas",
                  "min": 0.02, "max": 0.30, "default": 0.10},
    "threshold": {"description": "iso-level (lower = balls fuse into one goo, higher = tighter droplets)",
                  "min": 0.3, "max": 3.0, "default": 1.0},
    "edge_soft": {"description": "anti-aliased edge softness (smoothstep half-width)",
                  "min": 0.0, "max": 0.6, "default": 0.15},
    "color_mode": {"description": "how blobs are colored",
                   "choices": ["spectrum", "mono", "inferno", "goo"], "default": "spectrum"},
    "bg": {"description": "background color",
           "choices": ["black", "paper"], "default": "black"},
    "anim_mode": {"description": "animation mode (none=static, orbit=balls drift, pulse=radii breathe)",
                  "choices": ["none", "orbit", "pulse"], "default": "none"},
    "drift_amp": {"description": "orbit amplitude as fraction of canvas (orbit mode)",
                  "min": 0.0, "max": 0.3, "default": 0.12},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_metaballs(out_dir, seed: int, params=None):
    """Metaballs — implicit blobby/liquid surfaces (real-time CG classic).

    Technique: each ball contributes a radial potential F_i(p) = r_i^2 / |p - c_i|^2;
    the total field is F = Σ F_i. A pixel is "inside" when F ≥ threshold, producing
    the characteristic fused, liquid blobs that split as balls separate. Edges are
    anti-aliased with a smoothstep and each pixel is colored by the
    density-weighted blend of the balls' hues, giving a gooey gradient where
    blobs meet. First popularized in the 1980s demoscene / Pixar's "Knick Knack"
    (1989) and a staple of real-time implicit-surface rendering.

    With ``anim_mode="orbit"`` the balls drift along Lissajous paths and with
    ``anim_mode="pulse"`` their radii breathe — both are *structural* motion, so
    the result is genuinely live frame-to-frame (not a contrast-only flicker that
    a liveness metric would cull). ``anim_mode="none"`` is a true static baseline
    (Δ ≈ 0).
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        N = int(np.clip(params.get("balls", 8), 2, 16))
        ball_size = float(np.clip(params.get("ball_size", 0.10), 0.02, 0.30))
        threshold = float(np.clip(params.get("threshold", 1.0), 0.3, 3.0))
        edge_soft = float(np.clip(params.get("edge_soft", 0.15), 0.0, 0.6))
        color_mode = params.get("color_mode", "spectrum")
        bg = params.get("bg", "black")
        anim_mode = params.get("anim_mode", "none")
        drift_amp = float(np.clip(params.get("drift_amp", 0.12), 0.0, 0.3))
        anim_speed = float(np.clip(params.get("anim_speed", 1.0), 0.1, 3.0))

        _t = 0.0 if anim_mode == "none" else t * anim_speed

        cw, ch = get_canvas()
        W, H = int(cw), int(ch)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        Y = yy / H
        X = xx / W

        # ── per-ball base params (deterministic from seed) ──
        base_x = rng.random(N)
        base_y = rng.random(N)
        base_r = ball_size * (0.6 + 0.8 * rng.random(N))
        fx = 0.5 + 2.0 * rng.random(N)
        fy = 0.5 + 2.0 * rng.random(N)
        phx = rng.random(N) * 2.0 * math.pi
        phy = rng.random(N) * 2.0 * math.pi
        fr = 0.7 + 1.6 * rng.random(N)
        pr = rng.random(N) * 2.0 * math.pi
        hues = (rng.random(N) + np.arange(N) * 0.61803398875) % 1.0
        cols = np.array([_hsv_to_rgb(h, 0.85, 1.0) for h in hues])  # (N,3)

        # ── animated positions / radii ──
        if anim_mode == "orbit":
            cx = base_x + drift_amp * np.sin(_t * fx + phx)
            cy = base_y + drift_amp * np.cos(_t * fy + phy)
            rad = base_r
        elif anim_mode == "pulse":
            cx, cy = base_x, base_y
            rad = base_r * (1.0 + 0.45 * np.sin(_t * fr + pr))
        else:
            cx, cy = base_x, base_y
            rad = base_r

        # ── scalar field + density-weighted color ──
        F = np.zeros((H, W), dtype=np.float64)
        Cr = np.zeros((H, W), dtype=np.float64)
        Cg = np.zeros((H, W), dtype=np.float64)
        Cb = np.zeros((H, W), dtype=np.float64)
        eps = 1e-3
        for i in range(N):
            dx = X - cx[i]
            dy = Y - cy[i]
            w = (rad[i] * rad[i]) / (dx * dx + dy * dy + eps)
            F += w
            Cr += w * cols[i, 0]
            Cg += w * cols[i, 1]
            Cb += w * cols[i, 2]
        nonzero = F > 1e-4
        Cr = np.where(nonzero, Cr / np.where(nonzero, F, 1.0), 0.0)
        Cg = np.where(nonzero, Cg / np.where(nonzero, F, 1.0), 0.0)
        Cb = np.where(nonzero, Cb / np.where(nonzero, F, 1.0), 0.0)
        color_field = np.stack([Cr, Cg, Cb], axis=-1)

        if color_mode == "mono":
            color_field = np.stack([np.ones_like(F) * 0.9] * 3, axis=-1)
        elif color_mode == "inferno":
            color_field = _inferno(np.clip(F / (threshold * 2.0), 0.0, 1.0))
        elif color_mode == "goo":
            # greenish goo with brighter core
            g = np.clip(F / (threshold * 1.5), 0.0, 1.0)
            color_field = np.stack([0.2 * g, 0.4 + 0.6 * g, 0.3 * g], axis=-1)

        # ── composite with soft edge ──
        a = _smoothstep(threshold - edge_soft, threshold + edge_soft, F)
        if bg == "paper":
            bgc = np.array([0.97, 0.97, 0.97], dtype=np.float32)
        else:
            bgc = np.array([0.03, 0.03, 0.05], dtype=np.float32)
        rgb = bgc[None, None, :] + (color_field.astype(np.float32) - bgc[None, None, :]) * a[..., None]
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        # ── Provenance / fields (Rules 4, 5, 10) ──
        write_scalars(out_dir,
                      balls=N,
                      threshold=round(threshold, 2),
                      mean_field=round(float(F.mean()), 3),
                      inside_fraction=round(float((F >= threshold).mean()), 3))
        write_field(out_dir, np.clip(F / (threshold * 2.0), 0.0, 1.0).astype(np.float32))
        write_mask(out_dir, a.astype(np.float32))   # goo coverage mask

        capture_frame("505", rgb)
        save(rgb, mn(505, f"Metaballs t={_t:.2f}"), out_dir)
        return rgb
    except Exception as exc:
        cw, ch = get_canvas()
        fallback = np.ones((int(ch), int(cw), 3), dtype=np.float32) * 0.1
        save(fallback, mn(505, "Metaballs"), out_dir)
        print(f"[method_505] ERROR: {exc}")
        return fallback
