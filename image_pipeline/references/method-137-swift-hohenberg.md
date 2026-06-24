# Method 137 — Swift-Hohenberg Pattern Formation

- **Category:** simulations
- **ID:** 137
- **File:** `image_pipeline/methods/simulations/rayleigh_benard.py`
- **Function:** `sh()`

## Description

Single-field PDE producing Rayleigh-Bénard-like convection patterns:
hexagonal arrays, striped phases, localized spots, and spatiotemporal defect chaos.

The Swift-Hohenberg equation is the normal form of a finite-wavelength
pattern-forming instability (Turing bifurcation). It was originally derived
to describe thermal convection and produces the same iconic hexagonal
Bénard cells, roll patterns, and labyrinthine chaos — but with a single
scalar field that's guaranteed bounded by its -u³ saturation term.

## Physics

∂u/∂t = r·u - (∇² + q₀²)²·u - u³ + noise

- **r** — driving parameter, analogous to the Rayleigh number
  - r < 0: stable (no pattern)
  - 0 < r < 1: stripe patterns (supercritical pitchfork)
  - r > 1: hexagonal patterns, spots, chaos
  - r > 2: spatiotemporal chaos / defect turbulence

- **q₀** — preferred wavenumber (sets spatial scale of patterns)
  - Smaller q₀ = bigger patterns (q₀=0.05 gives ~100px features)
  - Larger q₀ = finer patterns (q₀=0.15 gives ~30px features)

- **-u³** — nonlinear saturation, guarantees bounded amplitude at any r

## Animation modes

| Mode | Default r | Default q₀ | Description |
|------|-----------|-------------|-------------|
| `evolve` | 1.5 | 0.08 | General pattern formation from noise |
| `hexagons` | 1.2 | 0.08 | Hexagonal honeycomb lattice |
| `stripes` | 2.5 | 0.10 | Labyrinthine stripe patterns |
| `spots` | 2.5 | 0.12 | Localized spot clusters |
| `chaos` | 3.5 | 0.08 | Spatiotemporal defect chaos |
| `obstacle` | 2.0 | 0.08 | Patterns around obstacles |

## Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `r` | 1.5 | -1.0 – 5.0 | Driving strength |
| `q0` | 0.08 | 0.02 – 0.30 | Preferred wavenumber |
| `dt` | 0.05 | 0.01 – 0.50 | Timestep (larger = faster per-frame change) |
| `noise` | 0.001 | 0 – 0.10 | Noise amplitude (sustains dynamics) |
| `grid_div` | 1 | 1 – 6 | Simulation grid divisor |

## Numerics

Spectral IMEX: implicit (∇² + q₀²)² in Fourier space, explicit -u³ + r·u
in physical space. 2/3 dealiasing rule. Bounded at any r — no clamps needed.

## Examples

```bash
# Hexagonal honeycomb lattice
uv run python -m image_pipeline.pipeline --method 137 \
  --params '{"anim_mode":"hexagons","r":1.2,"q0":0.08,"n_frames":300}' \
  --animate 137 --anim-duration 12

# Stripes with continuous dynamics
uv run python -m image_pipeline.pipeline --method 137 \
  --params '{"anim_mode":"stripes","r":2.5,"q0":0.10,"dt":0.1,"noise":0.005}' \
  --animate 137 --anim-duration 7.5

# Spatiotemporal chaos
uv run python -m image_pipeline.pipeline --method 137 \
  --params '{"anim_mode":"chaos","r":3.5,"q0":0.08,"n_frames":480,"noise":0.01}' \
  --animate 137 --anim-duration 20
```
