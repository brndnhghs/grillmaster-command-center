"""
#149 — Spectral Ocean Synthesis ("Storm Surge")

Procedural ocean surface synthesis using the JONSWAP/Pierson-Moskowitz
directional wave spectrum. The surface height field is constructed
in Fourier space each frame — phases advance according to the
deep-water dispersion relation, producing endlessly varying wave
patterns with realistic ocean physics.

Physics:
  h(x,y,t) = Re( Σ_k ĥ₀(k) · exp(i·(k·x - ω(k)·t + φ₀(k))) )

  ω(k) = √(g·|k|)          — deep-water dispersion (g=9.81)
  S(ω) ∝ ω⁻⁵·exp(-1.25(ω/ω_p)⁻⁴)·γ^exp(-(ω-ω_p)²/(2σ²ω_p²))  — JONSWAP
  D(θ) ∝ cos^(2s)(½(θ-θ_w))  — directional spreading

No integration step needed — the spectrum is evaluated analytically
per frame via IFFT. Generates satellite-level views of developing
ocean storms with wave groups, swell interference, and rogue waves.

Architecture A — single-call internal simulation with capture_frame().

Animation modes:
  evolve:     steady wind, waves propagate continuously
  storm_build: wind ramps up → sea state develops from calm to storm
  wind_shift:  wind direction rotates → crossing seas, diamond interference
  dual_swell:  two swell trains from different directions, beating patterns
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars
from ...core.animation import capture_frame


def _jonswap_spectrum(omega: np.ndarray, omega_p: float, gamma: float,
                      alpha: float) -> np.ndarray:
    """JONSWAP frequency spectrum S(ω).

    Args:
        omega: angular frequency array (rad/s)
        omega_p: peak frequency
        gamma: peak enhancement factor (1-7, 3.3 typical)
        alpha: Phillips constant (~0.0081 for fully-developed sea)
    Returns:
        S(ω) array, same shape as omega
    """
    g = 9.81
    # Sigma: width of peak
    sigma = np.where(omega <= omega_p, 0.07, 0.09)

    S = (alpha * g**2 * omega**(-5)
         * np.exp(-1.25 * (omega / omega_p)**(-4))
         * gamma**np.exp(-(omega - omega_p)**2 / (2 * sigma**2 * omega_p**2)))
    # Zero out DC and near-zero frequencies
    S[omega < 0.01] = 0.0
    S[~np.isfinite(S)] = 0.0
    return np.maximum(S, 0.0)


def _directional_spread(theta: np.ndarray, theta_wind: float,
                        s: float) -> np.ndarray:
    """Directional spreading D(θ) — cosine-squared power law.

    Args:
        theta: direction array (radians)
        theta_wind: mean wind direction (radians)
        s: spreading parameter (higher = narrower beam)
    Returns:
        D(θ) array, normalized to integrate to 1
    """
    dtheta = theta - theta_wind
    # Wrap to [-π, π]
    dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
    cos_term = np.cos(dtheta * 0.5)
    D = np.maximum(cos_term, 0.0) ** (2 * s)
    norm = np.sum(D) * (theta[0, 1] - theta[0, 0]) if D.shape[-1] > 1 else 1.0
    if norm > 0:
        D /= norm
    return D


def _render_height(h: np.ndarray) -> np.ndarray:
    """Render wave height — tanh sigmoid for dramatic wave fronts."""
    hc = h - np.mean(h)
    scale = max(abs(hc).max(), 0.01)
    h_norm = np.tanh(hc / scale * 2.5)
    return ((h_norm * 0.5 + 0.5) * 255).astype(np.uint8)


def _render_slope(hx: np.ndarray, hy: np.ndarray) -> np.ndarray:
    """Render slope magnitude |∇h| — shows wave steepness and breaking."""
    slope = np.sqrt(hx**2 + hy**2)
    scale = max(slope.max(), 0.001)
    s_norm = np.clip(slope / scale * 1.5, 0, 1)
    return (s_norm * 255).astype(np.uint8)


def _render_whitecap(h: np.ndarray, hx: np.ndarray, hy: np.ndarray,
                     threshold: float = 1.5) -> np.ndarray:
    """Render whitecap / breaking wave mask.

    Whitecaps occur where the slope exceeds a threshold (wave breaking).
    Paint those regions as bright white against darker wave heights.
    """
    hc = h - np.mean(h)
    scale = max(abs(hc).max(), 0.01)
    h_norm = np.tanh(hc / scale * 2.0)
    gray = ((h_norm * 0.3 + 0.3) * 255).astype(np.uint8)

    slope = np.sqrt(hx**2 + hy**2)
    s_scale = max(slope.max(), 0.001)
    s_norm = slope / s_scale
    # Whitecap mask: steep + positive height (crest)
    crest = hc > 0
    breaking = (s_norm > threshold) & crest
    gray[breaking] = 255
    return gray


# ═══════════════════════════════════════════════════════════════

@method(
    inputs={},
    id="167",
    name="Spectral Ocean Synthesis",
    category="simulations",
    tags=["animation", "ocean", "waves", "procedural", "fft",
           "jonswap", "storm", "natural"],
    timeout=300,
    params={
        "anim_mode": {
            "description": "wind / sea-state evolution mode",
            "choices": ["evolve", "storm_build", "wind_shift", "dual_swell"],
            "default": "evolve",
        },
        "wind_speed": {
            "description": "wind speed at 10m (m/s, 5-30)",
            "min": 3.0, "max": 35.0, "default": 15.0,
        },
        "fetch": {
            "description": "wind fetch (km, 10-200)",
            "min": 5.0, "max": 250.0, "default": 80.0,
        },
        "gamma": {
            "description": "JONSWAP peak enhancement γ (1.0-7.0, 3.3=typical sea)",
            "min": 1.0, "max": 7.0, "default": 3.3,
        },
        "wind_dir": {
            "description": "mean wind direction (degrees, 0-360)",
            "min": 0.0, "max": 360.0, "default": 45.0,
        },
        "spread": {
            "description": "directional spreading s (0.5-8.0, higher = narrower)",
            "min": 0.3, "max": 10.0, "default": 4.0,
        },
        "render_style": {
            "description": "wave field visualization",
            "choices": ["height", "slope", "whitecap"],
            "default": "height",
        },
        "scale": {
            "description": "meters per grid pixel (1=close-up wave detail, 20=satellite view)",
            "min": 0.5, "max": 30.0, "default": 4.0,
        },
        "n_frames": {
            "description": "simulation frames to capture",
            "min": 50, "max": 400, "default": 180,
        },
    },
    outputs={
        "image": "IMAGE",
        "luminance": "SCALAR",
        "wind_speed": "SCALAR",
        "peak_freq": "SCALAR",
        "significant_height": "SCALAR",
        "phillips_alpha": "SCALAR",
    },
)
def method_ocean_spectral(out_dir: Path, seed: int, params=None):
    """Spectral ocean wave synthesis via JONSWAP directional spectrum.

    Anim modes:
      evolve:     steady wind, waves propagate forever
      storm_build: wind ramps from calm to storm over the run
      wind_shift:  wind direction rotates continuously — crossing seas
      dual_swell:  two swell trains at 90° — interference diamonds
    Render: grayscale wave height (pipeline --recolor for color)
    """
    if params is None:
        params = {}

    anim_mode = str(params.get("anim_mode", "evolve"))
    wind_speed = float(params.get("wind_speed", 15.0))
    fetch = float(params.get("fetch", 80.0))
    gamma = float(params.get("gamma", 3.3))
    wind_dir_deg = float(params.get("wind_dir", 45.0))
    spread_s = float(params.get("spread", 4.0))
    render_style = str(params.get("render_style", "height"))
    scale_mpp = float(params.get("scale", 4.0))
    n_frames = int(params.get("n_frames", 180))

    rng = np.random.default_rng(seed)
    seed_all(seed)

    g = 9.81
    grid_div = 2
    sh, sw = H // grid_div, W // grid_div
    fh, fw = H, W

    # ── Spatial domain size (meters) ──
    # scale_mpp controls zoom: 1m/px = close-up waves, 20m/px = satellite view
    Lx = sw * scale_mpp
    Ly = sh * scale_mpp

    # ── Wave number grid ──
    kx = 2.0 * math.pi * np.fft.fftfreq(int(sw), d=scale_mpp)
    ky = 2.0 * math.pi * np.fft.fftfreq(int(sh), d=scale_mpp)
    kxx, kyy = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kxx**2 + kyy**2)
    omega = np.sqrt(g * k_mag)  # deep-water dispersion

    # ── Random initial phases ──
    noise_real = rng.normal(0, 1, (sh, sw)).astype(np.float64)
    noise_imag = rng.normal(0, 1, (sh, sw)).astype(np.float64)
    noise_fft = noise_real + 1j * noise_imag

    # ── Direction angles ──
    theta = np.arctan2(kyy, kxx)

    # ── Grid spacing in k-space ──
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]

    print(f"  Spectral Ocean | {anim_mode} wind={wind_speed:.0f}m/s "
          f"fetch={fetch:.0f}km γ={gamma:.2f} dir={wind_dir_deg:.0f}° "
          f"grid={sh}×{sw} L={Lx:.0f}m×{Ly:.0f}m")

    # ── Simulation loop ──
    for frame in range(n_frames):
        _t = frame / max(n_frames - 1, 1)
        t_phys = frame * 0.3  # physical time in seconds per frame

        # ── Mode-dependent parameter modulation ──
        if anim_mode == "evolve":
            U10 = wind_speed
            theta_wind = math.radians(wind_dir_deg)
            gamma_eff = gamma
        elif anim_mode == "storm_build":
            # Calm (U10=5) → storm (U10=30) over the run
            U10 = 5.0 + 28.0 * (_t**0.8)
            theta_wind = math.radians(wind_dir_deg)
            gamma_eff = gamma
        elif anim_mode == "wind_shift":
            U10 = wind_speed
            theta_wind = math.radians(wind_dir_deg + 180.0 * _t)
            gamma_eff = gamma
        elif anim_mode == "dual_swell":
            U10 = wind_speed
            theta_wind = math.radians(wind_dir_deg)
            gamma_eff = gamma
        else:
            U10 = wind_speed
            theta_wind = math.radians(wind_dir_deg)
            gamma_eff = gamma

        # ── JONSWAP peak frequency ──
        # ω_p = 22·(g/U10)·(U10²/(g·F))^0.33  (empirical fetch relation)
        F_m = fetch * 1000.0  # fetch in meters
        dimless_fetch = g * F_m / U10**2
        omega_p = 22.0 * (g / U10) * dimless_fetch**(-0.33)

        # Phillips constant α
        alpha = 0.076 * dimless_fetch**(-0.22)

        # ── Build Fourier spectrum ──
        S_w = _jonswap_spectrum(omega, omega_p, gamma_eff, alpha)

        # Directional spreading
        if anim_mode == "dual_swell":
            # Two swell trains at ±45° from wind
            D1 = _directional_spread(theta, theta_wind + math.radians(45), spread_s)
            D2 = _directional_spread(theta, theta_wind - math.radians(45), spread_s)
            D = (D1 + D2) * 0.5
        else:
            D = _directional_spread(theta, theta_wind, spread_s)

        # Full 2D spectrum
        spectrum_2d = S_w * D
        # Normalize spectrum energy
        spectrum_2d *= Lx * Ly / max(np.sum(spectrum_2d), 1e-20)

        # ── Fourier coefficients ĥ₀(k) = √(2·S(k)·dkx·dky) · noise ──
        dk = dkx * dky
        amplitude = np.sqrt(2.0 * spectrum_2d * dk)
        h0 = amplitude * noise_fft

        # ── Phase advance ──
        phase = omega * t_phys
        hk = h0 * np.exp(-1j * phase)  # -iωt: wave propagates in +x direction

        # IFFT to spatial domain
        h = np.real(np.fft.ifft2(hk)) * (sh * sw)

        # ── Compute gradient for slope/whitecap renders ──
        hx = np.real(np.fft.ifft2(1j * kxx * hk)) * (sh * sw)
        hy = np.real(np.fft.ifft2(1j * kyy * hk)) * (sh * sw)

        # ── Render ──
        if render_style == "slope":
            gray = _render_slope(hx, hy)
        elif render_style == "whitecap":
            gray = _render_whitecap(h, hx, hy, threshold=1.3)
        else:  # height
            gray = _render_height(h)

        canvas_np = np.stack([gray] * 3, axis=-1)
        canvas_img = Image.fromarray(canvas_np, mode="RGB")
        canvas_img = canvas_img.resize((fw, fh), Image.BILINEAR)
        canvas_np = np.array(canvas_img, dtype=np.uint8)

        save(canvas_np, f"frame_{frame:04d}.png", out_dir)
        capture_frame("167", canvas_np)

    write_scalars(out_dir,
                  wind_speed=U10,
                  peak_freq=float(omega_p),
                  significant_height=float(4.0 * np.std(h)),
                  phillips_alpha=float(alpha))

    print(f"  ✓ {n_frames} frames | h ∈ [{h.min():.3f}, {h.max():.3f}] "
          f"h_rms={np.std(h):.4f} Hs={4*np.std(h):.3f}m U10={U10:.1f}m/s "
          f"ω_p={omega_p:.3f} rad/s")
