# Method 138 — Swift-Hohenberg Pattern Formation

- **Category:** simulations
- **ID:** 138
- **File:** `image_pipeline/methods/simulations/rayleigh_benard.py`
- **Function:** `sh()`

## Description

Single-field PDE producing Rayleigh-Bénard-like convection patterns:
hexagonal arrays, striped phases, localized spots, and spatiotemporal defect chaos.
Includes spatial parameter sweep and temporal regime morphing modes.

∂u/∂t = r·u - (∇² + q₀²)²·u - u³ + noise

## Physics

- **r** — driving parameter, analogous to Rayleigh number
  - r < 0: stable (no pattern)
  - 0 < r < 1: stripe patterns
  - r > 1: hexagons, spots, chaos
  - r > 2: spatiotemporal defect turbulence

- **q₀** — preferred wavenumber (pattern scale)

- **-u³** — nonlinear saturation, guarantees bounded amplitude at any r

## Animation modes (8)

| Mode | r | Description |
|------|---|-------------|
| `evolve` | 1.5 | General pattern formation from noise |
| `hexagons` | 1.2 | Hexagonal honeycomb lattice |
| `stripes` | 2.5 | Labyrinthine stripe patterns |
| `spots` | 2.5 | Localized spot clusters (q₀=0.12) |
| `chaos` | 3.5 | Spatiotemporal defect chaos |
| `obstacle` | 2.0 | Patterns around staggered obstacles |
| `sweep` | 2.5 | r spatial gradient -0.5→4.0 across canvas (full regime spectrum) |
| `morph` | 2.0 | r oscillates 1.0→3.5 over time (continuous regime transitions) |

## Key Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `r` | 1.5 | -1.0 – 5.0 | Driving strength |
| `q0` | 0.08 | 0.02 – 0.30 | Preferred wavenumber |
| `dt` | 0.05 | 0.01 – 0.50 | Timestep |
| `noise` | 0.001 | 0 – 0.10 | Noise amplitude |
| `morph_speed` | 0.025 | 0.005 – 0.50 | Oscillation speed for sweep/morph |

## Numerics

Spectral IMEX with 2/3 dealiasing. Bounded at any r.

## Examples

```bash
# Spatial sweep (full regime spectrum)
uv run python -m image_pipeline.pipeline --method 138 \
  --params '{"anim_mode":"sweep","n_frames":360,"dt":0.06,"noise":0.004,"morph_speed":0.02}' \
  --animate 138 --anim-duration 15

# Temporal morph (regime transitions)
uv run python -m image_pipeline.pipeline --method 138 \
  --params '{"anim_mode":"morph","n_frames":480,"dt":0.06,"noise":0.005,"morph_speed":0.015}' \
  --animate 138 --anim-duration 20
```
