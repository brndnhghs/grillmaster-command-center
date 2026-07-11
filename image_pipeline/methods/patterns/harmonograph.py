from __future__ import annotations
import math

import numpy as np

from ...core.registry import method
from ...core.utils import save, mn, seed_all, write_field, write_scalars, W, H
from ...core.animation import capture_frame


def _gaussian_blur(buf: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (no scipy dependency)."""
    if sigma <= 0.15:
        return buf
    k = max(1, int(math.ceil(sigma * 3)))
    xs = np.arange(-k, k + 1, dtype=np.float64)
    ker = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    ker /= ker.sum()
    horiz = np.apply_along_axis(
        lambda row: np.convolve(row, ker, mode="same"), axis=1, arr=buf
    )
    vert = np.apply_along_axis(
        lambda col: np.convolve(col, ker, mode="same"), axis=0, arr=horiz
    )
    return vert


@method(
    id="406",
    name="Harmonograph",
    category="patterns",
    tags=["generative", "curve", "pendulum", "lissajous", "animation"],
    description=(
        "Lateral harmonograph: the damped superposition of four pendulum "
        "oscillations tracing a parametric curve. Classic 1844 apparatus, "
        "revived as generative art. Color encodes position along the curve."
    ),
    params={
        "freq1": {"description": "pendulum 1 frequency (x)", "min": 1.0, "max": 5.0, "default": 2.0},
        "freq2": {"description": "pendulum 2 frequency (x)", "min": 1.0, "max": 5.0, "default": 3.0},
        "freq3": {"description": "pendulum 3 frequency (y)", "min": 1.0, "max": 5.0, "default": 2.01},
        "freq4": {"description": "pendulum 4 frequency (y)", "min": 1.0, "max": 5.0, "default": 3.0},
        "damping": {"description": "pendulum decay (higher = tighter spiral)", "min": 0.0, "max": 0.02, "default": 0.004},
        "phase": {"description": "global phase offset", "min": 0.0, "max": 6.2832, "default": 0.0},
        "scale": {"description": "figure size as fraction of half-canvas", "min": 0.2, "max": 0.6, "default": 0.42},
        "line_width": {"description": "stroke softness / ribbon width (px)", "min": 0.3, "max": 4.0, "default": 1.2},
        "color_shift": {"description": "palette hue offset", "min": 0.0, "max": 1.0, "default": 0.5},
        "samples": {"description": "curve sample count (resolution)", "min": 20000, "max": 300000, "default": 120000},
        "anim_mode": {"description": "animation mode (none/phase/draw/rotate)", "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 3.0, "default": 1.0},
    },
)
def method_harmonograph(out_dir, seed: int, params=None):
    """Harmonograph — damped multi-pendulum parametric curve.

    A lateral harmonograph sums two decaying sinusoids per axis:

        x(t) = e^{-d t} ( sin(f1 t + p1) + sin(f2 t + p2) )
        y(t) = e^{-d t} ( sin(f3 t + p3) + sin(f4 t + p4) )

    Slightly detuned integer frequencies (e.g. 2.0 vs 2.01) make the figure
    precess slowly, producing the characteristic evolving rosette. The curve is
    rasterised into a density field, convolved with a small Gaussian to form a
    smooth ribbon, and tinted by an iq-style cosine palette indexed on the
    parameter t along the curve (so colour follows the pen).

    Animation modes (Architecture B — one frame per animation clock value):
      - none : static figure (seed-controlled via tiny phase jitter)
      - phase: pendulum phases drift with the clock -> the figure morphs
      - draw : the curve is progressively revealed along its length
      - rotate: the whole figure rotates about its centre
    """
    try:
        if params is None:
            params = {}
        t = float(params.get("time", 0.0))
        seed_all(seed)
        rng = np.random.default_rng(seed)

        f1 = float(params.get("freq1", 2.0))
        f2 = float(params.get("freq2", 3.0))
        f3 = float(params.get("freq3", 2.01))
        f4 = float(params.get("freq4", 3.0))
        damping = float(params.get("damping", 0.004))
        phase = float(params.get("phase", 0.0))
        scale = float(params.get("scale", 0.42))
        line_width = float(params.get("line_width", 1.2))
        color_shift = float(params.get("color_shift", 0.5))
        samples = int(params.get("samples", 120000))
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))

        # ── Animation clock ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # Seed-controlled tiny phase jitter so different seeds give different
        # figures, while remaining deterministic.
        jit = rng.uniform(0.0, 2.0 * math.pi, 4) * 0.08
        p1 = phase + jit[0]
        p2 = phase + math.pi / 2 + jit[1]
        p3 = phase + math.pi / 4 + jit[2]
        p4 = phase + math.pi * 0.75 + jit[3]

        if anim_mode == "phase":
            p1 += _t * 1.5
            p2 += _t * 2.0
            p3 += -_t * 1.3
            p4 += _t * 1.8

        # ── Parametric curve ──
        Tmax = 400.0
        tt = np.linspace(0.0, Tmax, samples)
        env = np.exp(-damping * tt)
        x = env * (np.sin(f1 * tt + p1) + np.sin(f2 * tt + p2))
        y = env * (np.sin(f3 * tt + p3) + np.sin(f4 * tt + p4))

        if anim_mode == "rotate":
            ca, sa = math.cos(_t * 1.5), math.sin(_t * 1.5)
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            x, y = xr, yr

        # Map to canvas pixels (curve spans ~[-2,2] before scale).
        px = x * (scale * 0.5 * W) + (0.5 * W)
        py = y * (scale * 0.5 * H) + (0.5 * H)

        # Progressive reveal for the "draw" mode.
        if anim_mode == "draw":
            progress = 0.1 + 0.9 * (0.5 - 0.5 * math.cos(_t))
            keep = (tt / Tmax) <= progress
            px = px[keep]
            py = py[keep]
            tt2 = tt[keep]
        else:
            tt2 = tt

        # ── Rasterise into a density field + time-along-curve field ──
        w_f, h_f = float(W), float(H)
        D, _, _ = np.histogram2d(
            px, py, bins=[W, H], range=[[0.0, w_f], [0.0, h_f]], weights=np.ones_like(px)
        )
        S, _, _ = np.histogram2d(
            px, py, bins=[W, H], range=[[0.0, w_f], [0.0, h_f]], weights=tt2
        )

        if line_width > 0.15:
            D = _gaussian_blur(D, line_width)
            S = _gaussian_blur(S, line_width)

        # Brightness: soft glow via sqrt of normalised density.
        peak = float(D.max())
        inten = np.zeros_like(D)
        if peak > 1e-6:
            inten = np.clip(D / peak, 0.0, 1.0)
            inten = np.sqrt(inten)

        # Colour: cosine palette indexed on normalised curve parameter.
        mean_t = np.zeros_like(D)
        nz = D > 1e-9
        mean_t[nz] = (S[nz] / D[nz]) / Tmax
        v = np.clip(mean_t + color_shift, 0.0, 1.0)
        r = 0.5 + 0.5 * np.cos(6.28318 * (1.0 * v + 0.0))
        g = 0.5 + 0.5 * np.cos(6.28318 * (0.75 * v + 0.33))
        b = 0.5 + 0.5 * np.cos(6.28318 * (0.5 * v + 0.66))
        rgb = np.stack([r, g, b], axis=-1) * inten[..., None]
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

        capture_frame("406", rgb)
        save(rgb, mn(406, "harmonograph"), out_dir)

        # ── Rules 4/5: write interesting scalars + the density field ──
        write_field(out_dir, D.astype(np.float32))
        write_scalars(
            out_dir,
            freq1=f1, freq2=f2, freq3=f3, freq4=f4,
            damping=damping, line_width=line_width,
            max_density=peak,
        )
        return rgb
    except Exception as exc:
        fallback = np.full((H, W, 3), 18, dtype=np.uint8)
        save(fallback, mn(406, "Harmonograph"), out_dir)
        print(f"[method_406] ERROR: {exc}")
        return fallback
