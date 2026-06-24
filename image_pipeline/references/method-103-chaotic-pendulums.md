# Method 103 — Chaotic Pendulums

**Research domain:** Classical chaos / butterfly effect / double pendulum dynamics
**Author:** Research-to-Method pipeline / Survey sub-agent

## Description

N double pendulums with nearly identical initial conditions diverge chaotically due to the butterfly effect. Each pendulum traces a colorful fading trail on a dark background, starting bunched together and gradually spreading into a complex, intertwined web.

The double pendulum is a classic chaotic system — even tiny differences in initial angle (0.001 rad) grow exponentially, producing wildly different trajectories after a few seconds. The visual narrative is the butterfly effect itself: near-identical starts → gradual divergence → full chaos.

## Algorithm

- **Physics:** Each pendulum follows the standard double pendulum Lagrangian equations (4 coupled ODEs: θ₁, θ₂, ω₁, ω₂), integrated with RK4 at dt=0.01 with 4 substeps per frame.
- **Initial conditions:** Pendulums start at θ₁≈1.8 rad (≈103°) and θ₂≈0.3 rad with initial angular velocity ω₁=2.5, ω₂=-1.8. Each pendulum gets a small random offset (±spread radians) to its initial angles.
- **Trails:** Each pendulum stores the last N tip positions. Trails are rendered as fading colored circles — oldest = dim/small, newest = bright/large.
- **Colors:** 36 pendulums distributed across the hue spectrum (HSV saturation=0.85, value=0.95) for maximum visual separation.
- **Butterfly effect:** With spread as low as 0.003 rad, pendulums start nearly on top of each other but diverge within 100-200 frames.

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| num_pendulums | 10–80 | 36 | Number of double pendulums |
| trail_length | 10–150 | 80 | Length of position trail per pendulum |
| spread | 0.0001–0.1 | 0.003 | Initial angle spread (radians) — smaller = more dramatic butterfly effect |
| n_frames | 100–600 | 300 | Number of simulation frames |

## Animation Modes

| Mode | Description | Visual effect |
|------|-------------|---------------|
| none | Static output | Final chaotic web after N frames |
| evolve | Internal simulation loop | Pendulums swing and diverge in real time |

**Architecture A** — no `"time"` param in invocation. Use:
```
--animate 103 --params '{"anim_mode":"evolve","n_frames":300,"spread":0.005}'
```

## Visual Characteristics

- Very dark blue-black background with vibrant colorful trails
- Pendulum arms drawn as thick colored lines connecting to bright glowing tips
- Trails fade from dim/small (oldest) to bright/large (newest) → neon paint-splatter aesthetic
- 36 colors span the full hue circle for maximum visual separation
- Output resolution: 768×512
- Deterministic per seed
- Speed: fast (~3s for 250 frames with 36 pendulums)

## Example Invocation

```bash
# Static render
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --methods 103 --force --no-cache

# Animation (Architecture A)
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 103 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":300,"num_pendulums":36,"spread":0.005}'

# Extreme butterfly effect (very small spread, many frames)
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 103 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":400,"num_pendulums":40,"spread":0.001,"trail_length":100}'

# Large spread for quick visible divergence
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 103 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":200,"num_pendulums":24,"spread":0.02}'
```

## Notes

- **Architecture A only.** Do NOT include `"time"` in params.
- RK4 integration with dt=0.01 ensures stability for the chaotic ODEs even at large initial angles.
- The butterfly effect is strongest with `spread=0.001-0.003` — pendulums start nearly identical and slowly diverge over 200+ frames.
- For a quicker visual, use `spread=0.01-0.02` — divergence is visible within 50 frames.
- Pendulum arm length is 120px. Origin is at canvas center (384, 256). The full swing spans ~300px vertically and ~250px horizontally.
- MP4 output typically ~100-200 KB (well under Discord's 10 MB limit).
