# Spectral Ocean Synthesis (#149)

JONSWAP/Pierson-Moskowitz directional wave spectrum reconstructed per frame via IFFT. No integration step — phases advance analytically.

## Physics

h(x,y,t) = Re( Σ_k ĥ₀(k) · exp(i·(k·x - ω(k)·t + φ₀(k))) )

- ω(k) = √(g·|k|) — deep-water dispersion
- S(ω) ∝ ω⁻⁵·exp(-1.25(ω/ω_p)⁻⁴)·γ^exp(...) — JONSWAP
- D(θ) ∝ cos^(2s)(½(θ-θ_w)) — directional spreading

## Parameters

| Param | Range | Default | Effect |
|-------|-------|---------|--------|
| wind_speed | 3-35 | 15 | m/s at 10m height |
| fetch | 5-250 | 80 | km wind fetch |
| gamma | 1-7 | 3.3 | JONSWAP peak enhancement |
| wind_dir | 0-360 | 45 | degrees wind direction |
| spread | 0.3-10 | 4 | directional spreading (higher=narrower) |
| scale | 0.5-30 | 4 | m per grid pixel (1=close-up, 20=satellite) |
| render_style | height/slope/whitecap | height | wave field viz |

## Animation Modes

| Mode | Visual |
|------|--------|
| evolve | Steady wind, waves propagate forever |
| storm_build | Wind ramps calm→storm over run |
| wind_shift | Wind rotates 180° — crossing seas |
| dual_swell | Two swell trains at ±45° — diamond interference |

## Notes

Pure FFT synthesis — no obstacles, no boundary conditions, no PDE solve. Open ocean only. Does what it says: endless realistic ocean wave animation from verified oceanographic spectra.
