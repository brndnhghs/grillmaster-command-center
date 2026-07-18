"""Allen–Cahn + Perona–Malik anisotropic diffusion (ID 146).

This is the **authoritative CPU export** for the GPU live-preview twin
(`acpm_seed` / `acpm_step` / `acpm_display` in core/shaders.py). The GPU twin
is a deliberately simplified live preview (it omits the per-frame noise and the
slow time-ramp on bias); this CPU node is the frame-accurate source of truth.

The model couples two classic image/phase-field equations on a single scalar
order parameter c(x) ∈ [-1, 1]:

  1. **Perona–Malik anisotropic diffusion** (Perona & Malik, IEEE PAMI 1990) —
     edge-preserving smoothing:

         ∂c/∂t = α · ∇·( g(|∇c|) ∇c ),   g(s) = 1 / (1 + (s/K)²)

     The conduction g is ≈1 on flat regions (ordinary diffusion) and →0 across
     strong edges, so structure is preserved while interiors smooth.

  2. **Allen–Cahn reaction** (Allen & Cahn, 1979) — non-conserving double-well
     relaxation:

         ∂c/∂t = c − c³ + bias

     c − c³ is the derivative of the symmetric double-well (minima at ±1); a
     positive `bias` favours the +1 phase, negative favours −1. Crucially this
     is **non-conserving** (unlike Cahn–Hilliard, node 951): the phase that
     wins simply grows, and small domains shrink and vanish — interfaces move by
     mean curvature.

Combined update (identical to the GPU twin's per-step formula):

    gx  = (c_r − c) / (1 + (c_r − c)² / K²)
    gxl = (c   − c_l) / (1 + (c   − c_l)² / K²)      (and y / up-down)
    diff = (gx − gxl) + (gy − gyl)
    c_new = c + dt · ( (c − c³ + bias) + α · diff )

The explicit 5-point scheme is stable because the PM flux → (Δc) for small
gradients (denominator → 1), so the effective diffusion coefficient is α and the
step limit is dt·α·4 < 2 — we auto-clamp dt to 0.45/α to stay unconditionally
stable across the whole param range.

Architecture A: a single call runs the internal time loop and captures each
frame via capture_frame(); the smooth temporal integration builds the MP4 in
memory (no discrete-time strobing, so no trail accumulation needed).

Reference: https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation
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
    write_field,
    write_mask,
    wired_source_lum,
)
from ...core.animation import capture_frame


# ── Colormaps (self-contained; match the diverging/inferno helpers elsewhere) ──
_CM = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.16, 0.04, 0.28],
        [0.47, 0.11, 0.42],
        [0.80, 0.32, 0.26],
        [0.98, 0.66, 0.13],
        [0.99, 0.99, 0.85],
    ],
    dtype=np.float64,
)


def _inferno(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    xq = t * 5.0
    i0 = np.minimum(xq.astype(np.int64), 4)
    f = xq - i0
    return _CM[i0] * (1.0 - f[..., None]) + _CM[i0 + 1] * f[..., None]


def _diverge(c: np.ndarray) -> np.ndarray:
    """Diverging colormap around c=0 (phase −1 ↔ phase +1)."""
    t = np.clip((c + 1.0) * 0.5, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    a = np.array([0.16, 0.18, 0.42], dtype=np.float64)
    b = np.array([0.98, 0.86, 0.62], dtype=np.float64)
    mid = np.array([0.92, 0.55, 0.30], dtype=np.float64)
    col = a[None, None, :] * (1.0 - s)[..., None] + b[None, None, :] * s[..., None]
    band = np.exp(-(c**2) / 0.02)[..., None]
    col = col + band * (mid - col) * 0.6
    return np.clip(col, 0.0, 1.0)


def _resize01(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    a = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((a * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


@method(
    id="146",
    name="AC + PM Diffusion",
    category="simulations",
    tags=[
        "phase-field",
        "allen-cahn",
        "perona-malik",
        "anisotropic-diffusion",
        "edge-preserving",
        "simulation",
        "procedural",
        "non-conserved",
    ],
    timeout=240,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "alpha": {
            "description": "Perona-Malik diffusion strength α — larger = faster edge-preserving smoothing (effective coeff; step auto-clamped for stability)",
            "min": 0.1,
            "max": 2.0,
            "default": 1.0,
        },
        "K": {
            "description": "PM conduction edge-sensitivity K — larger lets diffusion cross weaker edges (more abstracted)",
            "min": 0.02,
            "max": 1.0,
            "default": 0.2,
        },
        "bias": {
            "description": "Allen-Cahn constant double-well shift — + favours the +1 phase, − favours the −1 phase",
            "min": -0.5,
            "max": 0.5,
            "default": 0.0,
        },
        "dt": {
            "description": "time step (auto-clamped to 0.45/α for explicit stability)",
            "min": 0.005,
            "max": 0.4,
            "default": 0.1,
        },
        "init": {
            "description": "initial condition: noise, ±1 blobs, checkerboard, or seed from a wired image luminance",
            "choices": ["random", "blobs", "grid", "input_image"],
            "default": "random",
        },
        "colormap": {
            "description": "colour mapping of the order parameter c",
            "choices": ["grayscale", "diverge", "inferno"],
            "default": "grayscale",
        },
        "anim_mode": {
            "description": "evolution mode (evolve runs the PDE; freeze shows the initial condition)",
            "choices": ["evolve", "freeze"],
            "default": "evolve",
        },
        "anim_speed": {
            "description": "evolution rate (substeps per frame)",
            "min": 0.1,
            "max": 5.0,
            "default": 1.0,
        },
        "n_frames": {
            "description": "number of captured frames",
            "min": 30,
            "max": 400,
            "default": 90,
        },
    },
)
def method_ac_pm_diffusion(out_dir: Path, seed: int, params=None):
    """Allen–Cahn + Perona–Malik anisotropic diffusion — edge-preserving phase-field.

    A single non-conserved order parameter c self-organises into smooth domains
    separated by sharp, edge-preserving interfaces. Unlike the conserved
    Cahn–Hilliard node (951) the winning phase simply grows, so small domains
    shrink and vanish (mean-curvature motion of interfaces).

    Wiring: if an upstream IMAGE is wired and `init='input_image'`, its luminance
    seeds the initial c (bright → +phase, dark → −phase).
    """
    try:
        if params is None:
            params = {}

        seed_all(seed)
        rng = np.random.default_rng(seed)

        alpha = max(0.1, min(2.0, float(params.get("alpha", 1.0))))
        K = max(0.02, min(1.0, float(params.get("K", 0.2))))
        bias = max(-0.5, min(0.5, float(params.get("bias", 0.0))))
        dt_req = max(0.005, min(0.4, float(params.get("dt", 0.1))))
        # Explicit 5-pt stability: dt·α·4 < 2  →  dt < 0.5/α
        dt = min(dt_req, 0.45 / alpha)
        dt = max(1e-4, dt)

        init = str(params.get("init", "random"))
        colormap = str(params.get("colormap", "grayscale"))
        anim_mode = str(params.get("anim_mode", "evolve"))
        anim_speed = max(0.1, min(5.0, float(params.get("anim_speed", 1.0))))
        n_frames = int(params.get("n_frames", 90))

        w, h = int(W), int(H)
        N = 256  # square simulation lattice

        K2 = max(K * K, 1e-4)

        # ── Initial condition ──
        if init == "blobs":
            c = np.full((N, N), -1.0, dtype=np.float64)
            n_blobs = int(rng.integers(18, 32))
            yy, xx = np.ogrid[:N, :N]
            sign = 1.0
            for _ in range(n_blobs):
                cy = int(rng.integers(0, N))
                cx = int(rng.integers(0, N))
                r = int(rng.integers(N // 12, N // 6))
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                c[m] = sign
                sign = -sign
        elif init == "grid":
            blk = max(4, N // 12)
            cx = (np.arange(N) // blk) % 2
            c = ((cx[None, :] ^ cx[:, None]) * 2 - 1).astype(np.float64)
        elif init == "input_image":
            lum = wired_source_lum(params, N, N)
            if lum is not None:
                c = ((lum.astype(np.float64) * 2.0 - 1.0) * 0.9).clip(-0.95, 0.95)
            else:
                c = 0.1 * rng.standard_normal((N, N))
        else:  # random — coarsening from φ≈0 noise
            c = 0.1 * rng.standard_normal((N, N))

        # ── One explicit AC + PM step (periodic BC via np.roll) ──
        def _step(cc: np.ndarray) -> np.ndarray:
            cl = np.roll(cc, 1, axis=1)
            cr = np.roll(cc, -1, axis=1)
            cd = np.roll(cc, 1, axis=0)
            cu = np.roll(cc, -1, axis=0)
            gx = (cr - cc) / (1.0 + (cr - cc) ** 2 / K2)
            gy = (cu - cc) / (1.0 + (cu - cc) ** 2 / K2)
            gxl = (cc - cl) / (1.0 + (cc - cl) ** 2 / K2)
            gyl = (cc - cd) / (1.0 + (cc - cd) ** 2 / K2)
            diff = (gx - gxl) + (gy - gyl)
            ac = cc - cc**3 + bias
            nc = cc + dt * (ac + alpha * diff)
            return np.clip(nc, -1.5, 1.5)

        n_sub = max(1, int(round(anim_speed * 10)))

        def _frame_rgb(cur: np.ndarray) -> np.ndarray:
            if colormap == "inferno":
                t = (cur - cur.min()) / (cur.max() - cur.min() + 1e-8)
                return _inferno(t)
            elif colormap == "diverge":
                return _diverge(cur)
            else:  # grayscale — matches the GPU display twin exactly
                g = cur * 0.5 + 0.5
                return np.stack([g, g, g], axis=-1)

        last_img = None
        last_field = None
        last_mask = None

        for _frame in range(n_frames):
            if anim_mode == "evolve":
                for _ in range(n_sub):
                    c = _step(c)

            rgb = _frame_rgb(c)
            pil = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR
            )
            img = np.asarray(pil, dtype=np.float32) / 255.0
            last_img = img

            last_field = c.astype(np.float32)
            # mask = one of the two phases (c > 0)
            last_mask = (c > 0.0).astype(np.float32)

            capture_frame("146", img)

        if last_img is None:
            last_img = np.zeros((h, w, 3), dtype=np.float32)

        # ── Outputs (canvas resolution) ──
        field_out = _resize01(last_field, w, h) if last_field is not None else np.zeros(
            (h, w), dtype=np.float32
        )
        write_field(out_dir, field_out)

        mask_out = _resize01(last_mask, w, h) if last_mask is not None else np.zeros(
            (h, w), dtype=np.float32
        )
        write_mask(out_dir, mask_out)

        mean_c = float(c.mean())
        interface_frac = float(np.mean(np.abs(c) < 0.1))
        write_scalars(
            out_dir,
            mean_c=mean_c,
            interface_fraction=interface_frac,
            dt_effective=float(dt),
            dt_requested=float(dt_req),
            alpha=float(alpha),
            K=float(K),
            bias=float(bias),
            n_substeps=float(n_sub),
        )

        fname = mn(146, "ACPMDiffusion")
        save(last_img, fname, out_dir)
        return out_dir / fname
    except Exception as exc:  # Rule 1: PNG in every path
        fb = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fb, mn(146, "ACPMDiffusion"), out_dir)
        print(f"[method_146] ERROR: {exc}")
        return out_dir / mn(146, "ACPMDiffusion")
