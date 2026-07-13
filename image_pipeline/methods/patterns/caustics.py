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


def _spectral(v: np.ndarray) -> np.ndarray:
    """Inigo-Quilez cosine palette on a scalar in [0, 1]."""
    return 0.5 + 0.5 * np.cos(
        2.0 * math.pi * (v[:, :, None] * 0.8 + np.array([0.0, 0.33, 0.67])[None, None, :])
    )


@method(id="513", name="Caustics", category="patterns",
        new_image_contract=True,
        tags=["caustics", "water", "procedural", "refraction",
              "animation", "gpu-twin-candidate"],
        inputs={},
        outputs={"image": "IMAGE", "luminance": "FIELD"},
        params={
    "resolution": {"description": "internal evaluation grid (caustic is sampled at this many points per axis, then resized to canvas)", "min": 64, "max": 384, "default": 256},
    "depth": {"description": "water-column depth: the vertical distance the refracted ray travels to the floor (larger = more dramatic focusing)", "min": 0.2, "max": 3.0, "default": 1.0},
    "waves": {"description": "number of superimposed directional sine waves forming the water surface", "min": 2, "max": 7, "default": 4},
    "scale": {"description": "spatial frequency of the surface waves (higher = finer ripples)", "min": 1.0, "max": 20.0, "default": 6.0},
    "gain": {"description": "caustic brightness gain (how hot the focusing filaments get)", "min": 0.5, "max": 8.0, "default": 3.0},
    "colormode": {"description": "tint applied to the caustic filaments over the deep-water floor",
                  "choices": ["aqua", "gold", "mono", "spectral"], "default": "aqua"},
    "anim_mode": {"description": "animation mode (none/phase/drift)",
                  "choices": ["none", "phase", "drift"], "default": "none"},
    "anim_speed": {"description": "animation speed multiplier", "min": 0.1, "max": 5.0, "default": 1.0},
    "time": {"description": "animation phase [0, 2pi)", "min": 0.0, "max": 6.28, "default": 0.0},
})
def method_caustics(out_dir: Path, seed: int, params=None):
    """Caustics — animated underwater light focusing (Guardado & Sánchez-Crespo,
    *GPU Gems* Ch. 2 "Rendering Water Caustics", 2003).

    Caustics form when light refracts through a wavy water surface and focuses
    onto the floor in bright, curved filaments. The GPU Gems technique is a
    real-time simplification of backward Monte-Carlo ray tracing: for every
    point on the floor we project straight up to the wave surface, compute the
    surface normal, refract the (vertical) sun ray with Snell's law, and trace
    it back down to where it lands on the floor. Where many nearby rays
    converge, the floor is brightly lit — i.e. the caustic intensity is the
    *inverse area-magnification* of the floor-point → refracted-floor-hit map,
    which equals ``1 / |det J|`` with ``J`` the Jacobian of that map.

    This node reproduces that pipeline in closed form, fully vectorized:

      1. Sum-of-sines wave height ``H(u,v,t)`` above each floor point.
      2. Analytic surface normal ``N`` from the gradient (sin -> cos).
      3. Snell refraction of the downward sun ray (air→water, η = 1/1.33).
      4. Project the refracted ray down to the floor at ``depth`` D to obtain
         the displaced landing point ``(u', v')``.
      5. Jacobian ``J = d(u',v')/d(u,v)`` via finite differences; the caustic
         field is ``1/|det J|`` (bright where rays converge, clipped to a
         soft-saturated [0,1) so the floor stays dark between filaments).

    Closed-form per-frame field (Architecture B): the orchestrator re-calls the
    method with an increasing ``time``. The wave parameters (directions,
    frequencies, phases, speeds) are fixed by ``seed`` so the caustic *structure*
    is stable across frames and only the light pattern moves:

      * ``phase``  — each wave's temporal phase advances (smooth sweep);
      * ``drift``  — the wave phase advances *and* the sampling window pans, so
                     the whole caustic field slides across the floor;
      * ``none``   — pure function of the seed, static baseline (Δ ≈ 0) as
                     required by the audit.

    Animation is driven entirely by ``time`` through smooth (sin/linear) terms,
    so there are no cusps and ``none`` produces an identical frame every time.
    """
    try:
        if params is None:
            params = {}

        # ── Param extraction ──
        RES = int(params.get("resolution", 256))
        RES = max(64, min(384, RES))
        depth = float(params.get("depth", 1.0))
        n_waves = int(params.get("waves", 4))
        n_waves = max(2, min(7, n_waves))
        scale = float(params.get("scale", 6.0))
        gain = float(params.get("gain", 3.0))
        colormode = params.get("colormode", "aqua")
        anim_mode = params.get("anim_mode", "none")
        anim_speed = float(params.get("anim_speed", 1.0))
        t = float(params.get("time", 0.0))

        # ── Architecture-B time wiring ──
        _t = 0.0 if anim_mode == "none" else t * anim_speed

        # ── Seed wiring (fixed per seed; animation comes from time only) ──
        seed_all(seed)
        rng = np.random.default_rng(seed)

        # Wave components: seeded directions, frequencies, phases, temporal speeds
        angs = rng.uniform(0.0, 2.0 * math.pi, n_waves)
        freqs = rng.uniform(0.5, 1.5, n_waves) * scale
        amps = rng.uniform(0.4, 1.0, n_waves)
        phases = rng.uniform(0.0, 2.0 * math.pi, n_waves)
        speeds = rng.uniform(0.5, 1.5, n_waves)
        kx = freqs * np.cos(angs)   # (n_waves,)
        ky = freqs * np.sin(angs)

        # ── Floor grid (normalized coords, y down to match image rows) ──
        ys, xs = np.meshgrid(
            np.linspace(-1.0, 1.0, RES, dtype=np.float64),
            np.linspace(-1.0, 1.0, RES, dtype=np.float64),
            indexing="ij",
        )

        # Temporal phase advance (smooth)
        if anim_mode == "drift":
            # pan the sampling window as well as advancing the wave phase
            xs = xs + 0.30 * _t
            ys = ys + 0.15 * _t
            t_adv = _t
        else:
            t_adv = _t

        # ── Wave height + analytic gradient ──
        # phase_i(u,v) = kx_i*u + ky_i*v + speeds_i*t + phases_i   (broadcast over grid)
        # H = Σ a_i sin(phase_i);  ∂H/∂x = Σ a_i kx_i cos(phase_i);  ∂H/∂y = Σ a_i ky_i cos(phase_i)
        xu = xs[None, :, :] * kx[:, None, None]   # (n_waves, RES, RES)
        yv = ys[None, :, :] * ky[:, None, None]
        ph = xu + yv + speeds[:, None, None] * t_adv + phases[:, None, None]
        Hsum = np.tensordot(amps, np.sin(ph), axes=(0, 0))          # (RES, RES)
        dHdx = np.tensordot(amps * kx, np.cos(ph), axes=(0, 0))
        dHdy = np.tensordot(amps * ky, np.cos(ph), axes=(0, 0))

        # ── Surface normal (points up, z>0) ──
        Nx = -dHdx
        Ny = -dHdy
        Nz = np.ones_like(Nx)
        nrm = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
        Nx /= nrm
        Ny /= nrm
        Nz /= nrm

        # ── Snell refraction of the vertical sun ray L = (0,0,-1) ──
        # T = eta*I - (eta*dot(N,I) + sqrt(k)) * N ;  eta = n_air/n_water = 1/1.33
        eta = 1.0 / 1.33
        NdotL = -Nz                       # dot(N, (0,0,-1))
        k = 1.0 - eta * eta * (1.0 - NdotL * NdotL)
        k = np.maximum(k, 0.0)
        coef = eta * NdotL + np.sqrt(k)  # = -eta*Nz + sqrt(k)
        Tx = -coef * Nx
        Ty = -coef * Ny
        Tz = -eta - coef * Nz            # < 0 : ray travels downward

        # ── Project refracted ray to floor at depth D ──
        denom = np.maximum(-Tz, 1e-4)    # downward distance factor
        disp_x = Tx * (depth / denom)     # horizontal landing displacement
        disp_y = Ty * (depth / denom)
        # landing point x' = u + disp_x, y' = v + disp_y

        # ── Jacobian of the floor-point -> landing-point map ──
        # J = [[1+∂dx/∂x, ∂dx/∂y], [∂dy/∂x, 1+∂dy/∂y]]
        g_dx = np.gradient(disp_x)       # (d/dx along axis0(y), d/dx along axis1(x))
        g_dy = np.gradient(disp_y)
        Jxx = 1.0 + g_dx[1]              # ∂x'/∂x
        Jxy = g_dx[0]                   # ∂x'/∂y
        Jyx = g_dy[1]                   # ∂y'/∂x
        Jyy = 1.0 + g_dy[0]             # ∂y'/∂y
        detJ = Jxx * Jyy - Jxy * Jyx
        mag = np.abs(detJ)

        # Caustic intensity: inverse area magnification, baseline-subtracted
        # (flat surface -> mag = 1 -> 0 ; convergence -> mag -> 0 -> bright).
        caustic = (1.0 / mag - 1.0)
        caustic = np.clip(caustic, 0.0, None)
        caustic = caustic / (caustic + 1.0)        # soft-saturate to [0,1)
        caustic = np.power(caustic, 0.7)           # perceptual falloff
        caustic = np.clip(caustic * gain * 0.5, 0.0, 1.0)

        # ── Compose color (deep-water floor + tinted filaments) ──
        floor = np.array([0.02, 0.10, 0.16], dtype=np.float64)
        if colormode == "aqua":
            tint = np.array([0.55, 0.95, 1.0], dtype=np.float64)
        elif colormode == "gold":
            tint = np.array([1.0, 0.82, 0.40], dtype=np.float64)
        elif colormode == "mono":
            tint = np.array([0.95, 0.97, 1.0], dtype=np.float64)
        else:  # spectral
            tint = None
        c3 = caustic[:, :, None]
        if tint is None:
            rgb = (floor + c3 * (_spectral(caustic) * 1.4)).astype(np.float32)
        else:
            rgb = (floor + c3 * tint * 1.4).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        lum = rgb.mean(axis=-1)

        # ── Resize to canvas ──
        Hpx, Wpx = int(H), int(W)
        if (rgb.shape[0], rgb.shape[1]) != (Hpx, Wpx):
            arr = np.array(
                Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
                .resize((Wpx, Hpx), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0
            lum_full = np.array(
                Image.fromarray(((np.clip(lum, 0, 1) * 255).astype(np.uint8)))
                .resize((Wpx, Hpx), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0
        else:
            arr = rgb
            lum_full = lum

        # ── Provenance / fields (Rule 4 / Rule 5) ──
        write_scalars(out_dir,
                      mean=round(float(arr.mean()), 4),
                      std=round(float(arr.std()), 4),
                      peak=round(float(arr.max()), 4),
                      depth=round(float(depth), 3),
                      waves=n_waves,
                      gain=round(float(gain), 3))
        write_field(out_dir, lum_full.astype(np.float32))

        try:
            capture_frame("513", arr)
        except Exception:
            pass
        try:
            save(arr, mn(513, f"Caustics t={_t:.2f}"), out_dir)
        except Exception:
            pass
        return arr
    except Exception as exc:
        fallback = np.zeros((int(H), int(W), 3), dtype=np.float32)
        try:
            save(fallback, mn(513, "Caustics"), out_dir)
        except Exception:
            pass
        print(f"[method_513] ERROR: {exc}")
        return fallback
