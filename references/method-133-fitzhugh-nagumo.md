# FitzHugh-Nagumo Excitable Media (#133)

**Category:** simulations  
**File:** `image_pipeline/methods/simulations/fitzhugh_nagumo.py`

## Description

2-variable reaction-diffusion model of excitable media, modeling action potential
propagation in cardiac/neural tissue. Produces rotating spiral waves, concentric
target patterns, chaotic wave breaks, and meandering scroll waves.

## Physics

Two coupled fields evolve on a 768×512 grid:

- **u** — fast variable (membrane potential / excitation), ranges [-2, 2]
- **v** — slow variable (recovery current), ranges [-1, 1]

Barkley-scaled dynamics:
```
∂u/∂t = D_u·∇²u + (u - u³/3 - v) / ε
∂v/∂t = D_v·∇²v + ε·(u + a - b·v)
```

The 1/ε factor on u creates sharp wavefronts; the small ε on v makes recovery slow.
u is slaved to v, producing the characteristic excitable-medium dynamics.

### Key Parameters

| Param | Default | Range | Effect |
|-------|---------|-------|--------|
| `diff_u` | 1.5 | 0.1–5.0 | Wave speed scales as √(D_u/ε) |
| `diff_v` | 0.0 | 0.0–3.0 | Recovery diffusion; 0 gives sharpest spirals |
| `epsilon` | 0.12 | 0.01–0.5 | Timescale separation; smaller = slower recovery |
| `param_a` | 0.5 | 0.3–1.2 | Excitability threshold; lower = more excitable |
| `param_b` | 0.5 | 0.3–1.5 | Recovery rate; adjusts v-nullcline slope |
| `dt` | 0.08 | 0.02–0.5 | Sim timestep; affects wave speed |
| `n_frames` | 300 | 100–1200 | Total frames |
| `amplitude` | 1.0 | 0.2–2.0 | Initial perturbation amplitude |
| `render_style` | "u" | "u", "v", "uv_diff" | Which field to render as grayscale |

### Stability
- Diffusion CFL: D_u × dt < 0.5
- Reaction limit: dt × (1/ε) < 1.0 (clamp at ±3 prevents blowup)
- Sweet spot at defaults: D_u=1.5, ε=0.12, dt=0.08 → CFL=0.12, reaction step=0.67

## Animation Modes

| Mode | Description | Init |
|------|-------------|------|
| **spiral** | Rotating spiral wave from broken wavefront | Excited half-plane with circular gap |
| **target** | Concentric rings from oscillating central pacemaker | Rest state + point source at center |
| **chaos** | Multiple interacting waves with periodic sparks | Random patches + periodic perturbations |
| **scroll** | Meandering spiral tip via anisotropic diffusion | Broken wavefront, anisotropic D_u |
| **pacemaker** | Two competing pacemaker interference patterns | Two point sources at left and right |

## Rendering

u field mapped linearly to grayscale: (-2, 2) → (0, 255). Pipeline applies
`--recolor` for palette coloring. No edge enhancement needed — the Barkley
scaling produces naturally sharp wavefronts.

## Typical Use

```bash
# 15s spiral at default settings
python -m image_pipeline.pipeline --method 133 --params '{"anim_mode":"spiral","n_frames":360}' --animate 133 --anim-duration 15 --seed 42

# Target patterns
python -m image_pipeline.pipeline --method 133 --params '{"anim_mode":"target","n_frames":200}' --animate 133 --anim-duration 8 --seed 42

# Chaos mode
python -m image_pipeline.pipeline --method 133 --params '{"anim_mode":"chaos","n_frames":300}' --animate 133 --anim-duration 12 --seed 99

# Scroll wave
python -m image_pipeline.pipeline --method 133 --params '{"anim_mode":"scroll","n_frames":300}' --animate 133 --anim-duration 12 --seed 42

# Two pacemakers
python -m image_pipeline.pipeline --method 133 --params '{"anim_mode":"pacemaker","n_frames":200}' --animate 133 --anim-duration 8 --seed 42
```

## Implementation Notes

- Barkley scaling (1/ε on u reaction) creates sharp wavefronts essential for
  spiral formation — without it, wavefronts are too wide
- u and v clamped at [-3, 3] and [-2, 2] to prevent numerical blowup
- 5-point Laplacian via np.roll for all spatial derivatives
- Forward Euler time-stepping; no 1/ε on u diffusion term
- D_v=0 is standard for spiral waves — v evolves only through reaction term
