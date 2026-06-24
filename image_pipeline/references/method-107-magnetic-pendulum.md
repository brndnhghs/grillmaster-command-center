# Method 107 — Magnetic Pendulum

**Research domain:** Chaotic dynamics / fractal basin boundaries
**Author:** Research-to-Method pipeline / Survey sub-agent

## Description

A damped pendulum swings over a plane with 3 fixed magnets (red, green, blue). The pendulum's trail is colored by which magnet's basin of attraction it currently occupies. The boundaries between basins form intricate fractal filaments — a glowing lacework mandala.

The pendulum is modeled as a 2D harmonic oscillator with spring restoring force, Gaussian magnetic attraction, and velocity damping. RK4 integration with 12 substeps per frame.

## Algorithm

- **Physics:** 4th-order ODE system [x, y, vx, vy] with spring force (-k·r), velocity damping (-γ·v), and Gaussian magnetic attraction from 3 magnets in an equilateral triangle.
- **Integration:** RK4 at dt=0.01 with 12 substeps per frame.
- **Trail:** Position is sampled every other substep (~6 samples/frame), stored in a circular buffer. Trail length controls how many samples are visible.
- **Rendering:** Each trail point drawn as a glowing circle colored by closest magnet. Size and brightness fade with age.
- **Reset system:** Pendulum periodically resets to a new random position near center to explore different basin regions.

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| spring_k | 0.5–10.0 | 5.0 | Spring restoring force (higher = tighter to center) |
| damping | 0.0–0.5 | 0.04 | Velocity damping (higher = faster settling) |
| magnet_c | 0.1–5.0 | 0.8 | Magnet attraction strength |
| magnet_spread | 0.3–3.0 | 0.6 | Magnet distance from center |
| trail_length | 20–200 | 100 | Trail visibility length |
| reset_interval | 0–500 | 120 | Frames between resets (0=never) |

## Animation Modes

| Mode | Description |
|------|-------------|
| none | Static final frame |
| evolve | Internal simulation loop with trail building |

**Architecture A** — no `"time"` param. Use:
```
--animate 107 --params '{"anim_mode":"evolve","n_frames":300}'
```

## Visual Characteristics

- Dark background with red/green/blue glowing trail dots
- Three bright glowing magnet markers with pulsing cores
- White pendulum bob traces the current path
- Output resolution: 768×512
- Speed: fast (~3s for 300 frames)

## Example Invocation

```bash
# Animation
PYTHONPATH=... python3 -m image_pipeline.pipeline --animate 107 --force \
  --params '{"anim_mode":"evolve","n_frames":300,"trail_length":120,"reset_interval":150}'

# Quick exploration (shorter reset interval)
PYTHONPATH=... python3 -m image_pipeline.pipeline --animate 107 --force \
  --params '{"anim_mode":"evolve","n_frames":200,"reset_interval":50}'

# Stronger magnets for more chaotic behavior
PYTHONPATH=... python3 -m image_pipeline.pipeline --animate 107 --force \
  --params '{"anim_mode":"evolve","magnet_c":1.5,"spring_k":6.0}'
```
