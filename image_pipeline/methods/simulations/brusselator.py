from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

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
)
from ...core.animation import capture_frame


# 5-point discrete Laplacian; periodic boundaries handled by convolve2d wrap.
_LAP = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)

# Inferno-like colormap control points (t = 0..1).
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


def _resize01(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a float [0,1] 2D array to (h, w) preserving range as uint8."""
    a = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((a * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


@method(
    id="922",
    name="Brusselator",
    category="simulations",
    tags=[
        "reaction-diffusion",
        "pattern-formation",
        "turing",
        "spiral",
        "simulation",
        "procedural",
    ],
    inputs={},
    outputs={"image": "IMAGE", "field": "FIELD", "mask": "MASK"},
    params={
        "A": {
            "description": "Brusselator feed parameter A (sets the homogeneous steady state u*=A)",
            "min": 0.1,
            "max": 5.0,
            "default": 1.0,
        },
        "B": {
            "description": "Brusselator parameter B; B > 1 + A^2 gives the Turing / oscillatory regime",
            "min": 0.5,
            "max": 6.0,
            "default": 3.0,
        },
        "D_u": {
            "description": "diffusion coefficient of activator u",
            "min": 0.02,
            "max": 1.0,
            "default": 0.16,
        },
        "D_v": {
            "description": "diffusion coefficient of inhibitor v (D_v < D_u favours Turing patterns)",
            "min": 0.02,
            "max": 1.0,
            "default": 0.08,
        },
        "dt": {
            "description": "explicit-Euler time step per substep (stability: keep small)",
            "min": 0.01,
            "max": 0.2,
            "default": 0.05,
        },
        "init": {
            "description": "initial condition seeding the pattern",
            "choices": ["random", "seed-blob", "uniform", "oscillator"],
            "default": "random",
        },
        "show": {
            "description": "which concentration field to visualize",
            "choices": ["u", "v", "combined"],
            "default": "u",
        },
        "anim_mode": {
            "description": "evolution mode (evolve runs the PDE; freeze shows the initial state)",
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
def method_brusselator(out_dir, seed, params=None):
    """Brusselator — a classic two-species reaction-diffusion system.

    The Brusselator (Prigogine & Lefever, 1968) is one of the canonical
    models that exhibits a Turing instability: a homogeneous steady state
    (u* = A, v* = B/A) loses stability to spatial perturbations when the
    inhibitor diffuses faster than the activator (D_v < D_u) and B > 1 + A^2.
    The result is self-organizing spots, stripes and travelling spiral waves
    — the same mechanism invoked for biological morphogenesis and widely used
    in procedural texture / organic-pattern generation in computer graphics.

    The dynamics are integrated with explicit Euler on a square lattice:

        du/dt = A + u^2 v - (B + 1) u + D_u * Laplacian(u)
        dv/dt = B u - u^2 v       + D_v * Laplacian(v)

    Reference: https://en.wikipedia.org/wiki/Brusselator

    Architecture A: a single call runs the internal time loop and captures
    each frame. The smooth temporal integration means no trail accumulation is
    needed (no discrete-time strobing).
    """
    try:
        if params is None:
            params = {}

        seed_all(seed)
        rng = np.random.default_rng(seed)

        A = max(0.1, min(5.0, float(params.get("A", 1.0))))
        B = max(0.5, min(6.0, float(params.get("B", 3.0))))
        D_u = max(0.02, min(1.0, float(params.get("D_u", 0.16))))
        D_v = max(0.02, min(1.0, float(params.get("D_v", 0.08))))
        dt = max(0.01, min(0.2, float(params.get("dt", 0.05))))
        init = str(params.get("init", "random"))
        show = str(params.get("show", "u"))
        anim_mode = str(params.get("anim_mode", "evolve"))
        anim_speed = max(0.1, min(5.0, float(params.get("anim_speed", 1.0))))
        n_frames = int(params.get("n_frames", 90))

        w, h = int(W), int(H)
        N = 256  # square simulation lattice

        # ── Initial conditions (around the homogeneous steady state) ──
        if init == "seed-blob":
            u = np.full((N, N), A, dtype=np.float64)
            v = np.full((N, N), B / A, dtype=np.float64)
            cy = cx = N // 2
            r = max(2, N // 10)
            yy, xx = np.ogrid[:N, :N]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            u[m] = A + 0.6
            v[m] = B / A + 0.4
        elif init == "uniform":
            u = np.full((N, N), A, dtype=np.float64) + 0.01 * rng.standard_normal((N, N))
            v = np.full((N, N), B / A, dtype=np.float64) + 0.01 * rng.standard_normal((N, N))
        elif init == "oscillator":
            u = np.full((N, N), A, dtype=np.float64)
            v = np.full((N, N), B / A, dtype=np.float64)
            for _ in range(8):
                oy = int(rng.integers(0, N))
                ox = int(rng.integers(0, N))
                u[oy - 4 : oy + 4, ox - 4 : ox + 4] = A + 0.9
                v[oy - 4 : oy + 4, ox - 4 : ox + 4] = B / A + 0.6
        else:  # random
            u = A + 0.1 * rng.standard_normal((N, N))
            v = (B / A) + 0.1 * rng.standard_normal((N, N))
        u = np.clip(u, 0.0, None)
        v = np.clip(v, 0.0, None)

        def _step(uu, vv):
            Lu = convolve2d(uu, _LAP, mode="same", boundary="wrap")
            Lv = convolve2d(vv, _LAP, mode="same", boundary="wrap")
            du = A + uu * uu * vv - (B + 1.0) * uu + D_u * Lu
            dv = B * uu - uu * uu * vv + D_v * Lv
            nu = np.clip(uu + dt * du, 0.0, 20.0)
            nv = np.clip(vv + dt * dv, 0.0, 20.0)
            return nu, nv

        n_sub = max(1, int(round(anim_speed * 6)))

        def _frame_rgb(cur_u, cur_v):
            if show == "v":
                fld = cur_v
            elif show == "combined":
                fld = cur_u + cur_v
            else:
                fld = cur_u
            t = (fld - fld.min()) / (fld.max() - fld.min() + 1e-8)
            return _inferno(t)

        last_img = None
        last_mask = None
        last_field = None

        for frame in range(n_frames):
            if anim_mode == "evolve":
                for _ in range(n_sub):
                    u, v = _step(u, v)

            rgb = _frame_rgb(u, v)
            # upscale lattice to canvas
            pil = Image.fromarray(
                (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            ).resize((w, h), Image.BILINEAR)
            img = np.asarray(pil, dtype=np.float32) / 255.0
            last_img = img

            # mask = upper half of the activation range (structured regions)
            un = (u - u.min()) / (u.max() - u.min() + 1e-8)
            last_mask = (un > 0.5).astype(np.float32)
            last_field = u.astype(np.float32)

            capture_frame("922", img)

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

        write_scalars(
            out_dir,
            mean_u=float(u.mean()),
            max_u=float(u.max()),
            pattern_std=float(u.std()),
            n_substeps=float(n_sub),
        )

        fname = mn(922, "Brusselator")
        save(last_img, fname, out_dir)
        return out_dir / fname
    except Exception as exc:  # Rule 1: PNG in every path
        fb = np.full((int(H), int(W), 3), 0.5, dtype=np.float32)
        save(fb, mn(922, "Brusselator"), out_dir)
        print(f"[method_922] ERROR: {exc}")
        return out_dir / mn(922, "Brusselator")
