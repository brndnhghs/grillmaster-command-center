# Method 111 — Multi-Layer Rayleigh-Taylor Instability

**Author:** Research-to-Method (surprise mode expansion)  
**Category:** simulations  
**Tags:** physics, fluid, instability, cascade, animation

## Description

Three fluid layers with two independently-perturbed interfaces create cascading plume dynamics. The top interface (highest density contrast) grows first; its falling heavy plumes strike the middle layer, perturbing the second interface and triggering secondary instabilities below. The result is a rich cascade of interacting mushroom structures spanning the full canvas.

## Physics

- **3 density layers:** heavy (top, ρ=1.0) → medium (middle, ρ=middle_density) → light (bottom, ρ=0.0)
- **2 perturbed interfaces** with independent phase and frequency offsets
- Identical solver to #109/#110: Boussinesq + FFT Poisson + upwind advection + spectral low-pass
- Middle layer thickness and density are configurable

## Key Parameters

| Param | Range | Default | Effect |
|---|---|---|---|
| gravity | 0.1–5.0 | 1.2 | Buoyancy driving strength |
| perturb_freq | 1–6 | 2 | Base perturbation frequency |
| freq_offset | 0.0–4.0 | 1.5 | Frequency difference between layer interfaces |
| middle_density | 0.2–0.8 | 0.5 | Density of the middle layer |
| middle_height | 0.15–0.4 | 0.25 | Thickness ratio of the middle layer |
| sharpness | 4–24 | 10 | Interface sigmoid width |
| palette | ocean/fire/neon/plasma/moss/ice | ocean | Color scheme (6-stop gradient) |
| diffusion | 0.0–0.05 | 0.003 | Density diffusion rate |
| n_frames | 50–400 | 220 | Simulation frames |

## Animation Modes

- **evolve** — standard time evolution (both interfaces develop simultaneously with different growth rates)
- **palette_cycle** — sweeps through all 6 color palettes

## Visual Signature

**Early frames (0–60):** Two parallel wavy interfaces. The top interface develops visible mushroom plumes first.

**Middle frames (60–140):** Top plumes grow downward, elongating into the middle layer. Meanwhile, the second interface begins developing its own structures — often at different spatial frequencies due to the frequency offset.

**Late frames (140–200+):** Plumes from the top interface reach and impact the middle layer, distorting the second interface and creating complex secondary vortices. The bottom layer fills with a cascade of interacting plumes — some originating from the top interface, others seeded by the impact.

## Color Scheme

Uses a **6-stop gradient** mapped by density (unlike the 2-sigmoid scheme in #109/#110). Each density layer gets its own color range with smooth interpolation, plus sigmoid sharpening at the interfaces for crisp layer boundaries.

## Example Commands

```bash
# Default (ocean, 3 layers)
--animate 111 --params '{"gravity":1.2,"anim_mode":"evolve"}'

# Thick middle layer, different frequencies
--animate 111 --params '{"middle_height":0.35,"freq_offset":2.5,"palette":"plasma"}'

# Thin middle layer, high gravity
--animate 111 --params '{"middle_height":0.18,"gravity":2.0,"palette":"fire"}'

# Palette cycle
--animate 111 --params '{"anim_mode":"palette_cycle","palette":"ice"}'

# Close densities (subtle middle layer)
--animate 111 --params '{"middle_density":0.35,"gravity":1.5}'
```
