# Method 110 — Sheared Rayleigh-Taylor Instability

**Author:** Research-to-Method (surprise mode)  
**Category:** simulations  
**Tags:** physics, fluid, shear, instability, animation

## Description

A dense fluid with a horizontal velocity gradient (shear) sits above a stationary light fluid. The combination of gravitational instability (Rayleigh–Taylor) and shear-driven roll-up (Kelvin–Helmholtz) produces tilted, elongated, asymmetric mushroom plumes that sweep across the canvas.

## Physics

- **Boussinesq approximation** with vorticity-streamfunction formulation
- **FFT Poisson solver** for streamfunction
- **Upwind advection + spectral low-pass filtering** per frame
- **Shear profile:** horizontal velocity ramps linearly from 0 at the interface to `shear` at the top edge
- **Multi-frequency perturbation** (3 wave components) for richer plume structure

## Key Parameters

| Param | Range | Default | Effect |
|---|---|---|---|
| gravity | 0.1–5.0 | 1.0 | Buoyancy driving strength |
| shear | 0.0–8.0 | 2.0 | Horizontal shear velocity (0 = plain RT) |
| perturb_freq | 1–6 | 3 | Number of perturbation waves |
| atwood | 0.2–1.0 | 0.8 | Density contrast between fluids |
| sharpness | 4–24 | 12 | Interface sigmoid width |
| palette | ocean/fire/neon/plasma/moss | ocean | Color scheme |
| diffusion | 0.0–0.05 | 0.003 | Density diffusion rate |
| n_frames | 50–400 | 200 | Simulation frames |

## Animation Modes

- **evolve** — standard time evolution with drift in perturbation frequency
- **palette_cycle** — sweeps through all 5 color palettes
- **shear_burst** — oscillating shear strength (pulsing flow)

## Visual Signature

Tilted, swept-back mushroom plumes leaning in the direction of shear flow. At low shear (0.5–1.0): gently leaning plumes. At high shear (3.0–6.0): long, stretched, comet-like tails with trailing vortices. The asymmetry distinguishes this clearly from plain RT.

## Example Commands

```bash
# Default (ocean palette, moderate shear)
--animate 110 --params '{"gravity":1.0,"shear":2.0,"anim_mode":"evolve"}'

# Fire palette, high shear
--animate 110 --params '{"palette":"fire","shear":4.0,"perturb_freq":2}'

# Palette-cycle mode
--animate 110 --params '{"anim_mode":"palette_cycle","shear":2.0}'

# Neon with shear burst
--animate 110 --params '{"anim_mode":"shear_burst","palette":"neon","shear":3.0,"gravity":1.5}'

# Low Atwood (subtle contrast)
--animate 110 --params '{"atwood":0.4,"shear":1.5}'
```
