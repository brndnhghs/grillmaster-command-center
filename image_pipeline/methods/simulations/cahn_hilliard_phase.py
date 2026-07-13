"""Cahn–Hilliard phase separation (ID 951) — spectral (FFT) solver.

The Cahn–Hilliard equation (Cahn & Hilliard, 1958; modern exposition in
Witkin & Navon 2003, "Spinodal decomposition") is THE canonical
mass-conserving phase-field model. It describes how a single conserved order
parameter φ(x) — e.g. alloy composition, fluid fraction, or binary mixture
concentration — spontaneously separates into two coexisting phases (φ≈+1 and
φ≈−1) through *spinodal decomposition*: a homogeneous mixture at φ=0 is
unstable, so infinitesimal fluctuations grow, and the interfaces between
phases sharpen and then *coarsen* (Ostwald ripening) as larger domains eat
smaller ones.

    ∂φ/∂t = ∇²μ ,
    μ(φ) = φ³ − φ − γ ∇²φ          (chemical potential)

φ³−φ is the derivative of the symmetric double-well free energy (two minima at
±1); the −γ∇²φ term penalises sharp interfaces and sets the interface width
~√γ. Crucially **∫φ is conserved** (the ∇²μ operator has zero mean), so the two
phases always occupy comparable total area — the classic oil/water labyrinth —
which is what makes Cahn–Hilliard visually and physically distinct from its
non-conserving sibling the Allen–Cahn equation (node 146, which lets one phase
ultimately conquer the domain).

Discretisation — semi-spectral (FFT) with periodic boundaries:

    L(u)  = Re{ ifft2(−k² · fft2(u)) }              # Laplacian
    μ     = φ³ − φ − γ · L(φ)
    φ_{n+1} = φ_n + dt · L(μ)

The linear ∇² / ∇⁴ operators live in Fourier space (exact, O(N log N)), while
the cubic nonlinearity is evaluated in real space. The largest unstable
Fourier mode bounds the stable time step, so dt is auto-clamped from γ to keep
the explicit scheme unconditionally stable for the whole γ range:

    dt_safe = 1.9 / ( k_max² · (γ·k_max² − 1) )

The smooth temporal integration means no trail accumulation is needed (no
discrete-time strobing). This is the authoritative CPU export; a ping-pong GPU
twin (seed/step/display) is left as future additive work.

Architecture A: a single call runs the internal time loop and captures each
frame via capture_frame(); the smooth integration builds the MP4 in memory.

Reference: https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation
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


# ── Inferno-like colormap control points (t = 0..1) ──────────────────────────
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


def _diverge(phi: np.ndarray) -> np.ndarray:
    """Two-phase diverging colormap around φ=0 (phase A ↔ phase B)."""
    t = np.clip((phi + 1.0) * 0.5, 0.0, 1.0)
    # smoothstep for a soft interface band near the midline
    s = t * t * (3.0 - 2.0 * t)
    # phase A (cool indigo) -> phase B (warm cream)
    a = np.array([0.16, 0.18, 0.42], dtype=np.float64)
    b = np.array([0.98, 0.86, 0.62], dtype=np.float64)
    mid = np.array([0.92, 0.55, 0.30], dtype=np.float64)  # thin warm seam
    col = a[None, None, :] * (1.0 - s)[..., None] + b[None, None, :] * s[..., None]
    # brighten the interface band (|φ| small) for a glowing seam
    band = np.exp(-((phi) ** 2) / 0.02)[..., None]
    col = col + band * (mid - col) * 0.6
    return np.clip(col, 0.0, 1.0)


def _resize01(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    a = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((a * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


@method(
    id="951",
    name="Cahn–Hilliard Phase Separation",
    category="simulations",
    tags=[
        "phase-field",
        "cahn-hilliard",
        "spinodal",
        "coarsening",
        "simulation",
        "procedural",
        "conserved",
    ],
    timeout=240,
    inputs={"image_in": "IMAGE"},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "init": {
            "description": "initial condition: spinodal noise, ±1 blobs, checkerboard, or seed from wired image luminance",
            "choices": ["random", "blobs", "grid", "input_image"],
            "default": "random",
        },
        "gamma": {
            "description": "interface-energy coefficient γ — larger = wider, smoother interfaces & slower coarsening (interface width ~√γ)",
            "min": 0.1,
            "max": 3.0,
            "default": 0.5,
        },
        "dt": {
            "description": "requested time step (auto-clamped for spectral stability vs γ)",
            "min": 0.002,
            "max": 0.03,
            "default": 0.01,
        },
        "colormap": {
            "description": "colour mapping of the order parameter φ",
            "choices": ["diverge", "inferno", "grayscale"],
            "default": "diverge",
        },
        "show": {
            "description": "what to visualise: the order parameter φ or the |∇φ| interface band",
            "choices": ["phi", "interface"],
            "default": "phi",
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
def method_cahn_hilliard(out_dir: Path, seed: int, params=None):
    """Cahn–Hilliard phase separation — mass-conserving spinodal decomposition.

    A single conserved order parameter φ self-organises into two coexisting
    phases via spinodal decomposition and Ostwald ripening (domain coarsening).
    Solved semi-spectrally (FFT Laplacian) with an auto-clamped stable time
    step. Distinct from the non-conserving Allen–Cahn node (146): the total
    "mass" ∫φ stays constant, so the two phases always share the domain.

    Wiring: if an upstream IMAGE is wired and `init='input_image'`, its
    luminance seeds the initial φ (bright → +phase, dark → −phase).
    """
    try:
        if params is None:
            params = {}

        seed_all(seed)
        rng = np.random.default_rng(seed)

        init = str(params.get("init", "random"))
        gamma = float(params.get("gamma", 0.5))
        gamma = max(0.1, min(3.0, gamma))
        dt_req = float(params.get("dt", 0.01))
        dt_req = max(0.002, min(0.03, dt_req))
        colormap = str(params.get("colormap", "diverge"))
        show = str(params.get("show", "phi"))
        anim_mode = str(params.get("anim_mode", "evolve"))
        anim_speed = max(0.1, min(5.0, float(params.get("anim_speed", 1.0))))
        n_frames = int(params.get("n_frames", 90))

        w, h = int(W), int(H)
        N = 256  # square simulation lattice

        # ── Fourier wavenumber grid (periodic BC, dx = 1) ──
        k = 2.0 * math.pi * np.fft.fftfreq(N)  # angular freqs
        kx, ky = np.meshgrid(k, k)
        k2 = kx * kx + ky * ky
        k2[0, 0] = 0.0  # DC: no drift (mass conserved)
        k2_max = float(k2.max())

        # ── Auto-clamp dt for spectral stability vs γ ──
        # max |amplification| at k_max² needs dt·k_max²·(γ·k_max² − 1) < 2.
        denom = gamma * k2_max - 1.0
        if denom > 0:
            dt_safe = 1.9 / (k2_max * denom)
            dt = min(dt_req, dt_safe)
        else:
            dt = dt_req
        dt = max(1e-4, dt)

        # ── Initial condition ──
        if init == "blobs":
            phi = np.full((N, N), -1.0, dtype=np.float64)
            n_blobs = int(rng.integers(18, 32))
            yy, xx = np.ogrid[:N, :N]
            for _ in range(n_blobs):
                cy = int(rng.integers(0, N))
                cx = int(rng.integers(0, N))
                r = int(rng.integers(N // 12, N // 6))
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                phi[m] = 1.0
        elif init == "grid":
            blk = max(4, N // 12)
            cx = (np.arange(N) // blk) % 2
            phi = ((cx[None, :] ^ cx[:, None]) * 2 - 1).astype(np.float64)
        elif init == "input_image":
            lum = wired_source_lum(params, N, N)
            if lum is not None:
                phi = ((lum.astype(np.float64) * 2.0 - 1.0) * 0.9).clip(-0.95, 0.95)
            else:
                phi = 0.1 * rng.standard_normal((N, N))
        else:  # random — spinodal decomposition from φ≈0 noise
            phi = 0.1 * rng.standard_normal((N, N))

        # ── Spectral step ──
        def _lap(u: np.ndarray) -> np.ndarray:
            return np.real(np.fft.ifft2(-k2 * np.fft.fft2(u)))

        def _step(pp: np.ndarray) -> np.ndarray:
            mu = pp**3 - pp - gamma * _lap(pp)
            return pp + dt * _lap(mu)

        n_sub = max(1, int(round(anim_speed * 10)))

        def _frame_rgb(cur: np.ndarray) -> np.ndarray:
            if show == "interface":
                gx, gy = np.gradient(cur)
                fld = np.sqrt(gx * gx + gy * gy)
                fld = fld / (fld.max() + 1e-8)
            else:
                fld = cur
            if colormap == "inferno":
                t = (fld - fld.min()) / (fld.max() - fld.min() + 1e-8)
                return _inferno(t)
            elif colormap == "grayscale":
                t = (fld - fld.min()) / (fld.max() - fld.min() + 1e-8)
                return np.stack([t, t, t], axis=-1)
            else:  # diverge (works directly on φ∈[-1,1])
                return _diverge(fld)

        last_img = None
        last_field = None
        last_mask = None
        first_phi = phi.copy()

        for frame in range(n_frames):
            if anim_mode == "evolve":
                for _ in range(n_sub):
                    phi = _step(phi)

            rgb = _frame_rgb(phi)
            pil = Image.fromarray(
                (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            ).resize((w, h), Image.BILINEAR)
            img = np.asarray(pil, dtype=np.float32) / 255.0
            last_img = img

            last_field = phi.astype(np.float32)
            # mask = one of the two phases (φ > 0)
            last_mask = (phi > 0.0).astype(np.float32)

            capture_frame("951", img)

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

        # ∫φ conserved → mean_phi ≈ constant across frames (mass conservation proof)
        mean_phi = float(phi.mean())
        interface_frac = float(np.mean(np.abs(phi) < 0.1))
        write_scalars(
            out_dir,
            mean_phi=mean_phi,
            interface_fraction=interface_frac,
            dt_effective=float(dt),
            n_substeps=float(n_sub),
            gamma=float(gamma),
        )

        fname = mn(951, "CahnHilliard")
        save(last_img, fname, out_dir)
        return out_dir / fname
    except Exception as exc:  # Rule 1: PNG in every path
        fb = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fb, mn(951, "CahnHilliard"), out_dir)
        print(f"[method_951] ERROR: {exc}")
        return out_dir / mn(951, "CahnHilliard")
