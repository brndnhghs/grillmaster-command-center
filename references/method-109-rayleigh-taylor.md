# Method 109 — Rayleigh-Taylor Instability

**Author:** Research-to-Method (surprise mode)  
**Category:** simulations  
**Tags:** physics, fluid, instability, animation

## Description

Classic two-fluid Rayleigh–Taylor instability: a dense fluid layer above a lighter layer. Gravity amplifies a sinusoidal interface perturbation into broad mushroom-shaped plumes with Kelvin–Helmholtz roll-up at the edges. The Boussinesq approximation is solved with a vorticity-streamfunction formulation and FFT Poisson solver.

**Updated with:** Atwood number control, 6 color palettes, multi-frequency perturbation, palette cycle mode.

## Key Parameters

| Param | Range | Default | Effect |
|---|---|---|---|
| gravity | 0.1–5.0 | 1.0 | Buoyancy driving strength |
| perturb_freq | 1–6 | 3 | Number of perturbation waves |
| atwood | 0.2–1.0 | 0.8 | Density contrast (1.0 = max contrast) |
| sharpness | 4–24 | 12 | Interface sigmoid crispness |
| palette | ocean/fire/neon/plasma/moss/ice | ocean | Color scheme |
| diffusion | 0.0–0.05 | 0.003 | Density diffusion rate |
| n_frames | 50–400 | 220 | Simulation frames |

## Animation Modes

- **evolve** — standard time evolution (multi-frequency perturbation drifts)
- **palette_cycle** — sweeps through all 6 color palettes

## Visual Signature

Symmetric mushroom plumes growing downward from a horizontal interface. 3 wave components in the perturbation create richer plume shapes than a single sine wave. Atwood number controls the density contrast — low Atwood (0.3) gives subtle, diffuse plumes; high Atwood (0.9) gives sharp, dramatic mushroom caps.

## Example Commands

```bash
# Default: ocean palette, 3 waves, Atwood 0.8
--animate 109 --params '{"gravity":1.0,"anim_mode":"evolve"}'

# Fire palette, higher gravity
--animate 109 --params '{"palette":"fire","gravity":1.5,"atwood":0.9}'

# Low contrast, ethereal
--animate 109 --params '{"atwood":0.4,"palette":"plasma","sharpness":8}'

# Palette cycle
--animate 109 --params '{"anim_mode":"palette_cycle","gravity":1.2}'

# 1 perturbation wave (single large plume)
--animate 109 --params '{"perturb_freq":1,"gravity":1.5,"palette":"neon"}'

# 5 waves (many thin plumes)
--animate 109 --params '{"perturb_freq":5,"gravity":0.8,"palette":"moss"}'
```

## Related Methods

- **#110 Sheared RT** — adds horizontal shear flow for tilted, asymmetric plumes
- **#111 Multi-Layer RT** — 3 fluid layers with 2 cascading interfaces
