# Method 114 — Spring-Mass Network

**Research domain:** Physics simulation — deformable cloth/web  
**Author:** Research-to-Method pipeline (surprise survey)

## Description

A 2D grid of point masses connected by Hookean springs, simulated with Verlet integration. Starts with a crumpled initial state (Gaussian-filtered random height displacement) that creates organic fold patterns, then gently evolves under low gravity, wind, and optional sphere obstacle collision.

5 animation modes: billow, ripple, tornado, breathe, crumple.

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| grid_x | 20–100 | 55 | Horizontal mesh density |
| grid_y | 15–75 | 38 | Vertical mesh density |
| stiffness | 0.1–5.0 | 0.9 | Spring stiffness |
| damping | 0.85–0.999 | 0.97 | Velocity damping per frame |
| gravity | 0–500 | 15 | Gravity strength (low preserves crumpled folds) |
| wind_strength | 0–500 | 150 | Wind gust amplitude |
| color_by | height/stress/velocity | height | Coloring mode |
| palette | 10 options | crimson | Color palette |
| substeps | 1–20 | 4 | Physics substeps per frame |

## Animation Modes

| Mode | Description |
|------|-------------|
| billow | Low gravity + wind, crumpled folds persist and drift |
| ripple | Point impulses, wave propagation through mesh |
| tornado | Rotating vortex, fabric wraps around center |
| breathe | Rhythmic expansion/contraction |
| crumple | Starts heavily crumpled, slow relaxation |

## Example

```bash
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 114 --seed 42 --force \
  --params '{"anim_mode":"billow","n_frames":96,"gravity":15,"wind_strength":60,"palette":"crimson"}'
```

## Notes

- Low gravity (default 15) preserves the crumpled fold structure for more interesting visuals
- Crumpled initial state uses Gaussian-filtered noise (sigma=5.0) for coherent fold patterns
- For more dramatic deformation, increase gravity to 200+ and wind to 200+
- PAL rendering uses per-quad flat coloring via PIL ImageDraw — fine detail is limited by grid density
