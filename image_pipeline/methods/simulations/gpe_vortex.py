"""
#148 — GPE Quantum Vortex Turbulence

Gross-Pitaevskii Equation in a 2D trapped Bose-Einstein condensate.
A moving repulsive potential (stirring laser) nucleates quantized
vortices — topological defects where the phase winds by 2π around
a density null. Vortex-vortex interactions produce rich dynamics:
pairing, vortex streets, turbulent cascades, and dark solitons.

Physics:
  iℏ·∂ψ/∂t = (-ℏ²/2m·∇² + g|ψ|² + V_trap + V_stir)·ψ
  Split-step Fourier: half-step nonlinear → full-step kinetic
  Vortices detected via phase winding in 2×2 pixel loops

Animation modes:
  pairing:    two stirring lasers → vortex wakes & pairing
  turbulence: fast double-laser → dense vortex soup
  soliton:    phase-imprinted dark solitons (π phase jump)

Render styles:
  phase:      phase-coloured with saturation from density
  density:    grayscale density plot
  vortices:   phase-coloured + red vortex markers

Parameters:
  g:           nonlinearity (0.5-3.0), controls vortex core size
  alpha:       kinetic coefficient (0.1-1.0), controls healing length
  stir_amp:    stirring strength (2-15), taller = more vortices
  stir_speed:  stirring speed (0.1-1.0)
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H
from ...core.animation import capture_frame


# ═══════════════════════════════════════════════════════════════

@method(
    id="148",
    name="GPE Quantum Vortex Turbulence",
    category="simulations",
    tags=["animation", "quantum", "vortex", "fluid", "pde",
           "fft", "turbulence", "complex"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "stirring regime",
            "choices": ["pairing", "turbulence", "soliton"],
            "default": "pairing",
        },
        "render_style": {
            "description": "visualization style",
            "choices": ["phase", "density", "vortices"],
            "default": "phase",
        },
        "g": {
            "description": "nonlinearity (0.5-3.0)",
            "min": 0.2, "max": 4.0, "default": 1.0,
        },
        "alpha": {
            "description": "kinetic coefficient (0.1-1.0)",
            "min": 0.05, "max": 2.0, "default": 0.4,
        },
        "stir_amp": {
            "description": "stirring potential amplitude (2-15)",
            "min": 1, "max": 20, "default": 8,
        },
        "stir_speed": {
            "description": "stirring speed (0.1-1.0)",
            "min": 0.05, "max": 1.5, "default": 0.5,
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 100, "max": 600, "default": 300,
        },
    },
)
def method_gpe(out_dir: Path, seed: int, params=None):
    """GPE quantum vortex turbulence — quantized vortices in BEC.
    
    A rotating/oscillating repulsive stirrer nucleates quantized
    vortices in a 2D Bose-Einstein condensate. Phase-coloured 
    rendering reveals the complex wavefunction. Vortices appear
    as dark pinprick cores with phase winding.
    
    Anim modes:
      pairing:    two lasers → vortex wakes & pairing
      turbulence: fast double-laser → dense vortex soup
      soliton:    phase-imprinted dark solitons (π phase jump)

    Render styles:
      phase:      phase-coloured (HSV from complex argument)
      density:    grayscale density plot
      vortices:   phase-coloured + red vortex markers
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "pairing"))
    render_style = str(params.get("render_style", "phase"))
    g_val = float(params.get("g", 1.0))
    alpha = float(params.get("alpha", 0.4))
    stir_amp = float(params.get("stir_amp", 8))
    stir_speed = float(params.get("stir_speed", 0.5))
    n_frames = int(params.get("n_frames", 300))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    fh, fw = H, W
    
    # ── Precompute k-space ──
    kx = np.fft.fftfreq(fw).reshape(1, -1) * 2 * math.pi
    ky = np.fft.fftfreq(fh).reshape(-1, 1) * 2 * math.pi
    k2 = kx**2 + ky**2
    
    # ── Grid and trap ──
    yg, xg = np.mgrid[:fh, :fw]
    V_trap = ((xg - fw/2) / (fw * 0.3))**2 + ((yg - fh/2) / (fh * 0.35))**2
    
    # ── Initial wavefunction ──
    r2 = (xg - fw/2)**2 + (yg - fh/2)**2
    if anim_mode == "soliton":
        # Phase-imprinted: half the condensate has π phase offset
        psi = np.exp(-r2 / (fw * 0.18)**2).astype(np.complex128)
        psi[:, :fw//2] *= np.exp(1j * math.pi)
        psi += 0.02 * rng.standard_normal((fh, fw)) * (1 + 0j)
    else:
        psi = np.exp(-r2 / (fw * 0.2)**2) * (1 + 0.03 * rng.standard_normal((fh, fw)))
        psi = psi.astype(np.complex128)
    
    # Normalize
    psi = psi / np.sqrt(np.mean(np.abs(psi)**2))
    
    dt = 0.02
    print(f"  GPE Quantum Vortex | {anim_mode} g={g_val:.1f} α={alpha:.2f} "
          f"stir={stir_amp:.0f}@{stir_speed:.2f}")
    
    # Cache NPY for stir computation
    fw_f = float(fw)
    fh_f = float(fh)
    
    for frame in range(n_frames):
        t = frame * dt
        
        # ── Stirring potential ──
        if anim_mode == "pairing":
            # Two lasers, out of phase
            sx1 = 0.35 * fw_f + 0.12 * fw_f * math.sin(t * 0.3 * stir_speed)
            sy1 = 0.35 * fh_f + 0.12 * fh_f * math.cos(t * 0.25 * stir_speed)
            sx2 = 0.65 * fw_f + 0.12 * fw_f * math.sin(t * 0.3 * stir_speed + 0.7 * math.pi)
            sy2 = 0.65 * fh_f + 0.12 * fh_f * math.cos(t * 0.25 * stir_speed + 0.35 * math.pi)
            V_stir = (stir_amp * np.exp(-((xg - sx1)**2 + (yg - sy1)**2) / (fw_f * 0.025)**2)
                      + stir_amp * np.exp(-((xg - sx2)**2 + (yg - sy2)**2) / (fw_f * 0.025)**2))
        elif anim_mode == "turbulence":
            # Fast double-laser
            sx1 = fw_f/2 + 0.28 * fw_f * math.sin(t * 0.6 * stir_speed)
            sy1 = fh_f/2 + 0.22 * fh_f * math.cos(t * 0.5 * stir_speed)
            sx2 = fw_f/2 + 0.28 * fw_f * math.sin(t * 0.6 * stir_speed + math.pi)
            sy2 = fh_f/2 + 0.22 * fh_f * math.cos(t * 0.5 * stir_speed + math.pi)
            V_stir = (stir_amp * 1.5 * np.exp(-((xg - sx1)**2 + (yg - sy1)**2) / (fw_f * 0.025)**2)
                      + stir_amp * np.exp(-((xg - sx2)**2 + (yg - sy2)**2) / (fw_f * 0.02)**2))
        elif anim_mode == "soliton":
            # Single gentle stirrer
            sx = fw_f/2 + 0.15 * fw_f * math.sin(t * 0.15 * stir_speed)
            sy = fh_f/2 + 0.12 * fh_f * math.cos(t * 0.2 * stir_speed)
            V_stir = stir_amp * np.exp(-((xg - sx)**2 + (yg - sy)**2) / (fw_f * 0.02)**2)
        else:
            V_stir = 0.0
        
        V = V_trap + V_stir
        
        # ── GPE step: split-step Fourier ──
        # Half-step nonlinear
        dens = np.abs(psi)**2
        psi = psi * np.exp(-1j * (g_val * dens + V) * dt * 0.5)
        
        # Full-step kinetic in k-space
        psi = np.fft.ifft2(np.fft.fft2(psi) * np.exp(-1j * alpha * k2 * dt))
        
        # Half-step nonlinear
        dens = np.abs(psi)**2
        psi = psi * np.exp(-1j * (g_val * dens + V) * dt * 0.5)
        
        # Renormalize
        psi = psi / np.sqrt(np.mean(np.abs(psi)**2))
        
        # ── Render ──
        phase = np.angle(psi)
        dens = np.abs(psi)**2
        
        if render_style == "density":
            d_max = np.percentile(dens, 98) + 1e-10
            d_norm = np.clip(dens / d_max, 0, 1)
            img_gray = np.clip(d_norm * 255, 0, 255).astype(np.uint8)
            canvas_np = np.stack([img_gray] * 3, axis=-1)
        else:
            d_max = np.percentile(dens, 97) + 1e-10
            d_norm = np.clip(dens / d_max, 0, 1)
            
            # Phase → hue
            h = ((phase / (2 * math.pi) + 0.5) % 1.0)
            # Saturation: high at moderate density (vortex edges)
            sat = np.clip(1.0 - np.abs(d_norm - 0.5) * 2, 0, 1)
            # Value: bright at moderate density, dark at cores
            val = np.clip(d_norm * 1.2, 0, 1) * 0.85 + 0.15
            
            # HSV→RGB
            h6 = h * 6
            hi = h6.astype(np.int32) % 6
            f = h6 - hi
            p = val * (1 - sat)
            q = val * (1 - sat * f)
            t_hsv = val * (1 - sat * (1 - f))
            
            r = np.where(hi == 0, val, np.where(hi == 1, t_hsv, np.where(hi == 2, p, np.where(hi == 3, p, np.where(hi == 4, q, val)))))
            g_out = np.where(hi == 0, t_hsv, np.where(hi == 1, val, np.where(hi == 2, val, np.where(hi == 3, q, np.where(hi == 4, p, p)))))
            b = np.where(hi == 0, p, np.where(hi == 1, p, np.where(hi == 2, t_hsv, np.where(hi == 3, val, np.where(hi == 4, val, q)))))
            
            canvas_np = np.clip(np.stack([r, g_out, b], -1) * 255, 0, 255).astype(np.uint8)
            
            # Vortex markers for "vortices" render style
            if render_style == "vortices":
                dx = ((np.diff(phase, axis=1) + math.pi) % (2 * math.pi) - math.pi)
                dy = ((np.diff(phase, axis=0) + math.pi) % (2 * math.pi) - math.pi)
                wind = dx[:-1, :] + dy[:, :-1] - dx[1:, :] - dy[:, 1:]
                ys, xs = np.where(np.abs(wind) > math.pi * 0.5)
                # Draw red dots on vortex cores
                for vx, vy in zip(xs[:80], ys[:80]):
                    if 2 < vx < fw - 2 and 2 < vy < fh - 2:
                        canvas_np[vy-1:vy+2, vx-1:vx+2, 0] = 255
                        canvas_np[vy-1:vy+2, vx-1:vx+2, 1:] = 50
            
            # Subtle glow on bright regions
            lum = np.mean(canvas_np / 255.0, axis=-1).astype(np.float32)
            glow_img = Image.fromarray((lum * 255).astype(np.uint8))
            glow_arr = np.array(glow_img.filter(ImageFilter.GaussianBlur(radius=3))) / 255.0 * 0.05
            canvas_np = np.clip(canvas_np.astype(np.float32) + 
                                np.stack([glow_arr * 100, glow_arr * 80, glow_arr * 120], axis=-1),
                                0, 255).astype(np.uint8)
        
        save(canvas_np, mn(148, "GPE Quantum Vortex"), out_dir)
        capture_frame("148", canvas_np)
        
        if frame % 60 == 0:
            dens_mean = float(np.mean(np.abs(psi)**2))
            print(f"  {frame}/{n_frames}  |ψ|²={dens_mean:.2f}")
    
    print(f"  ✓ {n_frames} frames | {anim_mode} mode")
