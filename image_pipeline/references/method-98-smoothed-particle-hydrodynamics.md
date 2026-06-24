# Method 98 — Smoothed Particle Hydrodynamics

**Research domain:** Lagrangian fluid dynamics
**Author:** Research-to-Method pipeline / Survey sub-agent

## Description

2D Smoothed Particle Hydrodynamics (SPH) fluid simulation. ~1500 particles interact via SPH kernel functions (poly6 for density, spiky gradient for pressure, viscosity Laplacian for damping). Particles form a fluid pool that receives an initial velocity kick, creating a slosh wave that splashes against walls, develops vortex structures, and settles into complex motion patterns.

Rendering uses alpha-blended particle circles with Gaussian blur for smooth fluid surfaces. Three colouring schemes: velocity (viridis palette — dark purple to gold), dye (cyan vs magenta mixing), or both combined.

## Algorithm

- **Initialisation:** Particles arranged in a rectangular grid pool at the bottom of the canvas, with slight random perturbation. A horizontal velocity kick (stronger at the pool surface) creates the initial slosh.
- **SPH Loop (per frame):**
  1. **Density:** Sum of poly6 kernel weights over neighbours within smoothing radius `h`
  2. **Pressure:** Ideal gas law `p = k * (ρ - ρ₀)`, clamped to non-negative
  3. **Pressure force:** Symmetric pair-wise gradient of spiky kernel
  4. **Viscosity force:** Laplacian of viscosity kernel applied to velocity differences
  5. **Gravity:** Constant downward acceleration
  6. **Integration:** Semi-implicit Euler with wall collisions (damped reflection)
- **Rendering:** Each particle rendered as an RGBA circle with alpha proportional to velocity magnitude (faster = brighter). Gaussian blur (radius=3) smooths particle blobs into fluid surface.

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| num_particles | 500–3000 | 1500 | Number of fluid particles (higher = smoother fluid) |
| gravity_scale | 0.0–3.0 | 1.0 | Gravity multiplier (0 = zero-g float) |
| viscosity_scale | 0.0–3.0 | 1.0 | Fluid viscosity (higher = thicker, slower flow) |
| gas_scale | 0.1–5.0 | 1.0 | Gas stiffness (higher = more incompressible, bouncier) |
| render_mode | velocity/dye/both | velocity | Colouring scheme |
| n_frames | 50–300 | 150 | Number of simulation frames |

## Animation Modes

| Mode | Description | Visual effect |
|------|-------------|---------------|
| none | Static output | Final frame after 150 steps |
| evolve | Internal simulation loop | Slosh wave → wall splash → vortex formation → settling |

**Architecture A** — no `"time"` param in invocation. Use:
```
--animate 98 --params '{"anim_mode":"evolve","n_frames":150,"num_particles":1500}'
```

## Visual Characteristics

- Dark indigo background with velocity-coloured fluid (dark purple → teal → gold)
- Output resolution: 768×512
- Deterministic per seed; identical initial conditions produce identical fluid evolution
- Speed: moderate (~2-3s for 150 frames with 1500 particles)

## Example Invocation

```bash
# Static render
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --methods 98 --force --no-cache

# Animation (Architecture A)
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 98 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":150,"num_particles":1500,"render_mode":"velocity"}'

# Dye mixing mode
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 98 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":150,"num_particles":1500,"render_mode":"dye"}'

# Zero-gravity float
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 98 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":150,"num_particles":1500,"gravity_scale":0.0}'
```

## Notes

- **Architecture A only.** Do NOT include `"time"` in params — this triggers per-frame re-calling which restarts the simulation from scratch.
- Wall boundaries are damped (0.4 reflection coefficient) — particles lose energy on collision.
- Initial velocity kick is proportional to particle depth (stronger at surface), creating a realistic wave/slosh profile.
- Particles use semi-implicit Euler integration — stable for dt=0.003 with 1500 particles.
- Rendering uses alpha-blended circles + Gaussian blur for smooth fluid surface appearance.
- Best viewed with `render_mode="velocity"` for vortex structure visibility.
- For Discord delivery: MP4 ~200-500 KB (well under 10 MB limit).
