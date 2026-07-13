from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import (
    save, mn, seed_all, W, H, write_scalars, write_field,
)
from ...core.animation import capture_frame


# Inline inferno LUT (8 control stops, 0-255) -> normalized RGB
_INFERNO = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float64) / 255.0


def _inferno(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    x = t * 7.0
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, 7)
    f = x - i0
    return _INFERNO[i0] + (_INFERNO[i1] - _INFERNO[i0]) * f[..., None]


def _spectral(v: np.ndarray) -> np.ndarray:
    """Inigo-Quilez cosine palette on a scalar in [-1, 1]."""
    s = 0.5 + 0.5 * v
    return 0.5 + 0.5 * np.cos(
        2.0 * math.pi * s[:, :, None] + np.array([0.0, 0.33, 0.67])[None, None, :]
    )


@method(id="512", name="SIREN Field", category="patterns",
        new_image_contract=True,
        tags=["implicit-neural", "siren", "procedural", "texture",
              "animation", "gpu-twin-candidate"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "resolution": {"description": "internal evaluation grid (SIREN samples this many points per axis)", "min": 64, "max": 384, "default": 256},
    "hidden": {"description": "hidden-layer width of the periodic MLP", "min": 16, "max": 256, "default": 64},
    "layers": {"description": "number of hidden layers", "min": 1, "max": 6, "default": 3},
    "omega0": {"description": "first-layer frequency (SIREN's signature high-frequency input scaling)", "min": 1, "max": 60, "default": 30},
    "omega": {"description": "frequency of subsequent hidden layers", "min": 1, "max": 60, "default": 30},
    "weight_scale": {"description": "weight-init standard deviation multiplier (controls detail/contrast)", "min": 0.1, "max": 3.0, "default": 1.0},
    "coord_scale": {"description": "input-coordinate multiplier (higher = more repetitions / zoom-out)", "min": 0.5, "max": 12.0, "default": 3.0},
    "colormode": {"description": "how the 3 output neurons are mapped to color",
                  "choices": ["rgb", "grayscale", "spectral", "inferno"], "default": "rgb"},
    "anim_mode": {"description": "animation mode (none/phase/rotate/translate/freq_sweep)",
                  "choices": ["none", "phase", "rotate", "translate", "freq_sweep"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_siren(out_dir: Path, seed: int, params=None):
    """SIREN Field — an Implicit Neural Representation sampled as an image.

    Technique (Sitzmann, Martel, Berg, Lindell & Wetzstein, "Implicit Neural
    Representations with Periodic Activation Functions", NeurIPS 2020). A
    SIREN is a multilayer perceptron whose activations are *sine* functions:

        φ₀(x) = x
        φᵢ(x) = sin(ωᵢ · (Wᵢ φᵢ₋₁(x) + bᵢ))     for i ≥ 1
        y(x)  = W_out φ_L(x) + b_out

    The first layer uses a high frequency ω₀ (default 30) so the network can
    represent fine spatial detail; later layers use ω. We evaluate this MLP over
    a regular grid of (x, y) coordinates to produce an RGB image — i.e. the
    image *is* the neural field's value at each pixel.

    With randomly initialized (untrained) weights a SIREN already yields rich,
    organic, quasi-periodic interference textures: a superposition of Gabor-like
    basis functions at random orientations and frequencies (the network's
    structural prior, described in the paper). The ``seed`` controls the weight
    initialization, so every seed is a distinct emergent pattern; ``weight_scale``
    and ``coord_scale`` further sculpt detail and repetition.

    Closed-form per-frame field (Architecture B): the orchestrator re-calls the
    method with an increasing ``time``. Animation modes modulate the field
    smoothly (no cusps):
      * ``phase``       — a global phase offset sweeps the interference pattern;
      * ``rotate``      — input coordinates rotate about the center;
      * ``translate``   — the sampling window pans across the field;
      * ``freq_sweep``  — ω₀ oscillates (smooth sine), morphing feature scale.
    With ``anim_mode=\"none\"`` the field is a pure function of the seed, so it
    is a static baseline (Δ ≈ 0) as required.
    """
    try:
        if params is None:
            params = {}

        # ── Param extraction ──
        RES = int(params.get("resolution", 256))
        RES = max(64, min(384, RES))
        hidden = int(params.get("hidden", 64))
        hidden = max(16, min(256, hidden))
        layers = int(params.get("layers", 3))
        layers = max(1, min(6, layers))
        omega0 = float(params.get("omega0", 30.0))
        omega = float(params.get("omega", 30.0))
        weight_scale = float(params.get("weight_scale", 1.0))
        coord_scale = float(params.get("coord_scale", 3.0))
        colormode = params.get("colormode", "rgb")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))

        # ── Architecture-B time wiring ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Seed wiring ──
        seed_all(seed)
        rng = np.random.default_rng(seed)

        # ── Input coordinates in [-1, 1] × coord_scale ──
        ys, xs = np.meshgrid(
            np.linspace(-1.0, 1.0, RES, dtype=np.float64),
            np.linspace(-1.0, 1.0, RES, dtype=np.float64),
            indexing="ij",
        )
        # animate the sampling window
        if anim_mode == "rotate":
            ang = _t
            ca, sa = math.cos(ang), math.sin(ang)
            xr = xs * ca - ys * sa
            yr = xs * sa + ys * ca
            xs, ys = xr, yr
        elif anim_mode == "translate":
            xs = xs + _t * 0.6
            ys = ys + _t * 0.25
        coords = np.stack([xs, ys], axis=-1) * coord_scale  # (RES, RES, 2)
        pts = coords.reshape(-1, 2)  # (N, 2)

        # ── SIREN weight initialization (Sitzmann et al. 2020) ──
        # First layer: c0 = sqrt(6 / fan_in) / ω0 ; hidden/output: c = sqrt(6 / fan_in).
        def _u(fan_in, c):
            return (rng.random((fan_in, hidden)) * 2.0 - 1.0) * c * weight_scale

        def _ub(fan_in):
            c = math.sqrt(6.0 / max(1, fan_in))
            return (rng.random((fan_in,)) * 2.0 - 1.0) * c * weight_scale

        c0 = math.sqrt(6.0 / 2.0) / max(1e-3, omega0)
        W1 = _u(2, c0)
        b1 = _ub(hidden) * 0.5
        Ws, bs = [], []
        for _ in range(layers - 1):
            ch = math.sqrt(6.0 / hidden)
            Ws.append(_u(hidden, ch))
            bs.append(_ub(hidden) * 0.5)
        Wo = (rng.random((hidden, 3)) * 2.0 - 1.0) * math.sqrt(6.0 / hidden) * weight_scale
        bo = (rng.random((3,)) * 2.0 - 1.0) * weight_scale

        # ── Forward pass (vectorized over all grid points) ──
        h = np.sin(omega0 * (pts @ W1 + b1))  # first layer uses ω₀
        for Wl, bl in zip(Ws, bs):
            h = np.sin(omega * (h @ Wl + bl))
        if anim_mode == "freq_sweep":
            # smoothly oscillate the input frequency (no cusp)
            om = omega0 * (1.0 + 0.5 * math.sin(_t))
            h = np.sin(om * (pts @ W1 + b1))
            for Wl, bl in zip(Ws, bs):
                h = np.sin(omega * (h @ Wl + bl))
        out = np.tanh(h @ Wo + bo)  # (N, 3) in [-1, 1]

        # ── Color mapping ──
        out3 = out.reshape(RES, RES, 3)
        if colormode == "rgb":
            rgb = 0.5 + 0.5 * out3
            lum = out3.mean(axis=-1)
        elif colormode == "grayscale":
            lum = out3.mean(axis=-1)
            g = np.clip(0.5 + 0.5 * lum, 0.0, 1.0)
            rgb = np.stack([g, g, g], axis=-1)
        elif colormode == "spectral":
            lum = out3.mean(axis=-1)
            rgb = _spectral(lum)
        else:  # inferno
            lum = out3.mean(axis=-1)
            rgb = _inferno(0.5 + 0.5 * lum)
        rgb = rgb.astype(np.float32)

        # phase animation is applied as a pre-activation shift on the first layer
        if anim_mode == "phase":
            h2 = np.sin(omega0 * (pts @ W1 + b1) + _t)
            for Wl, bl in zip(Ws, bs):
                h2 = np.sin(omega * (h2 @ Wl + bl))
            out2 = np.tanh(h2 @ Wo + bo).reshape(RES, RES, 3)
            if colormode == "rgb":
                rgb = (0.5 + 0.5 * out2).astype(np.float32)
            else:
                lum2 = out2.mean(axis=-1)
                if colormode == "grayscale":
                    g = np.clip(0.5 + 0.5 * lum2, 0.0, 1.0)
                    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
                elif colormode == "spectral":
                    rgb = _spectral(lum2).astype(np.float32)
                else:
                    rgb = _inferno(0.5 + 0.5 * lum2).astype(np.float32)

        # ── Resize to canvas ──
        Hpx, Wpx = int(H), int(W)
        if (rgb.shape[0], rgb.shape[1]) != (Hpx, Wpx):
            arr = np.array(
                Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
                .resize((Wpx, Hpx), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0
            lum_full = np.array(
                Image.fromarray(((np.clip(lum, -1, 1) * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
                .resize((Wpx, Hpx), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0
        else:
            arr = rgb
            lum_full = np.clip(lum, -1, 1) * 0.5 + 0.5

        # ── Provenance / fields (Rule 4 / Rule 5) ──
        write_scalars(out_dir,
                      mean=round(float(arr.mean()), 4),
                      std=round(float(arr.std()), 4),
                      peak=round(float(arr.max()), 4),
                      omega0=round(float(omega0), 2),
                      hidden=hidden,
                      layers=layers)
        write_field(out_dir, lum_full.astype(np.float32))

        try:
            capture_frame("512", arr)
        except Exception:
            pass
        try:
            save(arr, mn(512, f"SIREN Field t={_t:.2f}"), out_dir)
        except Exception:
            pass
        return arr
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        try:
            save(fallback, mn(512, "SIREN Field"), out_dir)
        except Exception:
            pass
        print(f"[method_512] ERROR: {exc}")
        return fallback
