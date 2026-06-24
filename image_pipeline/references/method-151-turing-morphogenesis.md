# Turing Morphogenesis (ID 151)

Schnakenberg reaction-diffusion on a growing/shrinking domain. Spectral semi-implicit integration. Patterns stretch and bifurcate as the domain expands, or compress as it shrinks. Center-crop zoom in/out.

**File:** `methods/simulations/turing_morphogenesis.py`
**Function:** `turing_morphogenesis`

## Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| a | 0.05–0.5 | 0.1 | Schnakenberg parameter a |
| b | 0.5–2.0 | 0.9 | Schnakenberg parameter b |
| Du | 0.001–0.1 | 0.01 | Activator diffusion |
| Dv | 0.1–2.0 | 0.5 | Inhibitor diffusion |
| gamma | 2–100 | 30 | Reaction rate |
| growth_rate | -0.02–0.02 | 0.005 | Domain expansion (negative = zoom out) |
| noise | 0–0.1 | 0.005 | Sustained noise |
| mode | spots/stripes/labyrinth/mixed | mixed | Initial seed pattern |
| n_frames | 100–1500 | 480 | Frame count |
| grid_div | 1–4 | 2 | Coarse grid factor |
| dt | 0.001–0.05 | 0.01 | Time step |

## Modes

- **spots** — localized spots grow into Turing patterns (γ=12, growth=0.003)
- **stripes** — directional stripes stretch and bifurcate (γ=12, growth=0.003)
- **labyrinth** — random noise → labyrinthine patterns (γ=30, growth=0.005)
- **mixed** — spots + noise hybrid (γ=12, growth=0.003)

## Growth Rate Guide

| growth_rate | Effect | Scale at 480f |
|-------------|--------|---------------|
| 0.005 | Strong zoom in | 1.25 |
| 0.002 | Moderate zoom in | 1.09 |
| 0.001 | Gentle zoom in | 1.05 |
| 0 | Static domain | 1.00 |
| -0.0005 | Gentle zoom out | 0.98 |
| -0.004 | ~1px/frame zoom out | 0.84 |

## Notes

- Spectral semi-implicit integration for stability
- Center-crop for zoom in, center-pad with steady state for zoom out
- Sustained noise keeps patterns alive on static/slow domains
- Clamped to [-10, 10] to prevent blowup
- Labyrinth mode is most stable at high gamma
