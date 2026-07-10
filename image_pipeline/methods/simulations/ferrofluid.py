"""
#149 — Ferrofluid Rosensweig Instability

A ferrofluid (magnetic colloidal suspension) under an oscillating
perpendicular magnetic field. Above a critical field strength, the
flat free surface destabilizes into sharp spikes (Rosensweig
instability). Spatially varying B fields create traveling wave
fronts, radial ripples, and competing wave patterns.

Physics:
  ∂h/∂t = B²(x,y)·f(h) − γ·∇⁴h − α·h + η·∇²h + lateral_drift
  Semi-implicit spectral solver: explicit driving → implicit smoothing
  Height-dependent feedback: peaks concentrate field, grow taller
  Tanh saturation limits spike height naturally

Animation modes:
  sweep-h:       traveling horizontal wave bands
  sweep-diag:    diagonal traveling wave front
  ripple-ring:   concentric radial ripple from center
  dual-waves:    competing horizontal & vertical bands
  pulse-clean:   uniform pulse (one clean cycle)
  pulse-wiggle:  uniform pulse with lateral wiggle during decay

Parameters:
  gamma:    surface tension (0.01-0.1), controls spike spacing
  alpha:    gravity/damping (0.01-0.1), controls collapse speed
  eta:      viscosity (0.1-0.5), controls smoothing rate
  speed:    overall timescale multiplier (0.5-2.0)
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
    id="149",
    name="Ferrofluid Rosensweig Instability",
    category="simulations",
    tags=["animation", "ferrofluid", "magnetic", "instability",
           "spikes", "pde", "spectral", "labyrinth"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "B-field pattern",
            "choices": ["sweep-h", "sweep-diag", "ripple-ring", "dual-waves",
                        "pulse-clean", "pulse-wiggle"],
            "default": "sweep-h",
        },
        "gamma": {
            "description": "surface tension (0.01-0.1)",
            "min": 0.005, "max": 0.2, "default": 0.03,
        },
        "alpha": {
            "description": "gravity/damping (0.01-0.1)",
            "min": 0.005, "max": 0.2, "default": 0.05,
        },
        "eta": {
            "description": "viscosity (0.1-0.5)",
            "min": 0.05, "max": 1.0, "default": 0.2,
        },
        "speed": {
            "description": "timescale multiplier (0.5-2.0)",
            "min": 0.3, "max": 3.0, "default": 1.0,
        },
        "n_frames": {
            "description": "simulation frames",
            "min": 100, "max": 600, "default": 300,
        },
    },
)
def method_ferrofluid(out_dir: Path, seed: int, params=None):
    """Ferrofluid Rosensweig instability — magnetic spike formation.
    
    Height field h(x,y) driven by spatially varying magnetic field.
    Semi-implicit spectral PDE solver. Spikes erupt where local B
    exceeds critical threshold. Traveling wave fronts sweep across
    the canvas.
    
    Anim modes:
      sweep-h:       horizontal traveling wave bands
      sweep-diag:    diagonal traveling wave front
      ripple-ring:   concentric radial ripple from center
      dual-waves:    competing horizontal & vertical bands
      pulse-clean:   uniform pulse (one clean cycle)
      pulse-wiggle:  uniform pulse with lateral wiggle
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "sweep-h"))
    gamma = float(params.get("gamma", 0.03))
    alpha = float(params.get("alpha", 0.05))
    eta = float(params.get("eta", 0.2))
    speed = float(params.get("speed", 1.0))
    n_frames = int(params.get("n_frames", 300))

    seed_all(seed)
    fh, fw = H, W
    
    # ── Precompute k-space ──
    kx = np.fft.fftfreq(int(fw)).reshape(1, -1) * 2 * math.pi
    ky = np.fft.fftfreq(int(fh)).reshape(-1, 1) * 2 * math.pi
    k2 = kx**2 + ky**2
    k4 = k2**2
    
    # ── Coordinate grid for B-field patterns ──
    yg, xg = np.mgrid[:fh, :fw]
    Wf, Hf = float(fw), float(fh)
    
    # ── Initial height: flat + tiny noise ──
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((fh, fw), dtype=np.float32) * 0.002
    
    B_critical = 0.6
    print(f"  Ferrofluid | {anim_mode} γ={gamma:.3f} α={alpha:.3f} η={eta:.2f} speed={speed:.2f}")
    
    for frame in range(n_frames):
        t = frame * 0.03 * speed
        
        # ── Magnetic field pattern ──
        B_amp = 2.5 * max(0, math.sin(t * 0.3))**2
        
        if anim_mode == "sweep-h":
            B_mod = 0.7 + 0.3 * np.cos(2 * math.pi * xg / Wf * 2 - t * 0.3)
        elif anim_mode == "sweep-diag":
            B_mod = 0.7 + 0.3 * np.cos(2 * math.pi * (xg/1.5 + yg/1.0) / max(Wf, Hf) * 1.5 - t * 0.4)
        elif anim_mode == "ripple-ring":
            r_grid = np.sqrt((xg - Wf/2)**2 + (yg - Hf/2)**2)
            B_mod = 0.7 + 0.3 * np.cos(2 * math.pi * r_grid / (Wf * 0.15) - t * 0.6)
        elif anim_mode == "dual-waves":
            B_mod = 0.6 + 0.2 * np.cos(2 * math.pi * xg / Wf * 3 - t * 0.5) \
                        + 0.2 * np.cos(2 * math.pi * yg / Hf * 2 - t * 0.3)
        elif anim_mode == "pulse-clean":
            B_mod = 1.0  # uniform
        elif anim_mode == "pulse-wiggle":
            B_mod = 1.0
        else:
            B_mod = 1.0
        
        B_map = np.clip(B_amp * B_mod, 0, 3).astype(np.float32)
        
        # ── Lateral field (in-plane) ──
        if anim_mode == "pulse-wiggle":
            Bx = float(1.8 * math.sin(t * 0.02) * max(0, math.sin(t * 0.3)))
            By = float(1.2 * math.cos(t * 0.018) * max(0, math.sin(t * 0.3)))
        else:
            Bx = 1.0 * math.sin(t * 0.012)
            By = 0.6 * math.cos(t * 0.01)
        
        # ── PDE step: semi-implicit ──
        rf = 1.0 + 5.0 * np.clip(1.0 - B_map / B_critical, 0, 1)
        eta_eff = eta * rf
        alpha_eff = alpha * rf
        
        h_mean = float(np.mean(h))
        ds = np.clip(B_map - B_critical, 0, None) * 4.0
        sat = np.clip(1.0 - h**2 / 4.0, 0, 1.5)
        h_gain = np.tanh((h - h_mean) * 3.0)
        driving = ds * h_gain * sat
        
        hk = np.fft.fft2(h.astype(np.float64))
        dhx = np.fft.ifft2(1j * kx * hk).real.astype(np.float32)
        dhy = np.fft.ifft2(1j * ky * hk).real.astype(np.float32)
        lateral = -Bx * dhx - By * dhy
        
        noise = rng.standard_normal((fh, fw), dtype=np.float32) * 0.002
        
        h_ex = h + 0.3 * (driving + lateral + noise)
        hk = np.fft.fft2(h_ex.astype(np.float64))
        denom = np.float64(1.0 + 0.3 * (eta_eff * k2 + gamma * k4 + alpha_eff))
        h = np.fft.ifft2(hk / denom).real.astype(np.float32)
        h = np.tanh(h * 0.4) * 3.0
        
        # ── Render ──
        h_clip = np.clip(h, -3.0, 3.0)
        h_norm = (h_clip + 3.0) / 6.0
        bright = np.clip(h_norm * 1.5 - 0.3, 0, 1)
        
        hk_r = np.fft.fft2(h.astype(np.float64))
        dhx_r = np.fft.ifft2(1j * kx * hk_r).real.astype(np.float32)
        dhy_r = np.fft.ifft2(1j * ky * hk_r).real.astype(np.float32)
        grad = np.sqrt(dhx_r**2 + dhy_r**2)
        g_norm = np.clip(grad / (np.percentile(grad, 88) + 1e-6), 0, 1)
        angle = np.arctan2(dhy_r, dhx_r) / (2 * math.pi) + 0.5
        
        r_ch = np.zeros_like(h) + 0.01
        g_ch = np.zeros_like(h) + 0.01
        b_ch = np.zeros_like(h) + 0.04
        
        iri = g_norm * np.clip(1.0 - bright * 1.8, 0, 1)
        r_ch += iri * (0.6 + 0.4 * np.sin(angle * 4 + 0.3))
        g_ch += iri * (0.3 + 0.3 * np.sin(angle * 4 + 1.5))
        b_ch += iri * (0.1 + 0.2 * np.cos(angle * 4))
        
        pk = bright ** 0.5
        r_ch += pk * 0.90
        g_ch += pk * 0.92
        b_ch += pk * 0.98
        
        canvas_np = np.clip(np.stack([r_ch, g_ch, b_ch], -1) * 255, 0, 255).astype(np.uint8)
        
        lum = np.mean(canvas_np / 255.0, axis=-1).astype(np.float32)
        glow_img = Image.fromarray((lum * 255).astype(np.uint8))
        glow_arr = np.array(glow_img.filter(ImageFilter.GaussianBlur(radius=3))) / 255.0 * 0.04
        canvas_np = np.clip(canvas_np.astype(np.float32) + 
                           np.stack([glow_arr * 120, glow_arr * 100, glow_arr * 80], axis=-1),
                           0, 255).astype(np.uint8)
        
        save(canvas_np, mn(149, "Ferrofluid"), out_dir)
        capture_frame("149", canvas_np)
        
        if frame % 60 == 0:
            print(f"  {frame}/{n_frames}  h∈[{h.min():.2f},{h.max():.2f}]  "
                  f"B∈[{B_map.min():.2f},{B_map.max():.2f}]")
    
    print(f"  ✓ {n_frames} frames | {anim_mode} mode")
