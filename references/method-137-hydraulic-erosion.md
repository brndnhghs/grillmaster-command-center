# Method 137 — Hydraulic Erosion / River Network Terrain

| Field | Value |
|---|---|
| **ID** | 137 |
| **Category** | simulations |
| **File** | `image_pipeline/methods/simulations/hydraulic_erosion.py` |
| **Architecture** | A — internal simulation loop with `capture_frame()` |
| **Timeout** | 300s |

## Description

Couples water flow, sediment transport, and terrain evolution on a 2D height field. Water falls as rain, flows downhill via steepest-descent routing, erodes sediment proportional to stream power, and deposits when transport capacity drops. Thermal weathering smooths steep slopes above the angle of repose. Creates continuously evolving drainage networks with Horton's law scaling — rills deepen into tributaries, meanders develop, and alluvial fans pulse outward.

## Animation Modes

| Mode | Description |
|---|---|
| `hydraulic` | Rainfall + flowing water erosion carves branching river networks (default) |
| `thermal` | Slope-dependent thermal smoothing only — rounds peaks, fills valleys, no water |
| `combined` | Both hydraulic erosion and thermal weathering active together |
| `tectonic` | Continuous uplift + hydraulic erosion → persistent river incision and canyon formation |
| `coastal` | Wave erosion along left edge (cliff retreat) + beach deposition on right edge |
| `none` | Static snapshot of initial fractal terrain |

## Key Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `rain_rate` | 0.008 | 0–0.05 | Rainfall per cell per frame — higher = more runoff |
| `K_e` | 0.05 | 0.001–0.5 | Erosion coefficient (stream power) — higher = faster incision |
| `K_d` | 0.1 | 0.01–1.0 | Deposition coefficient — higher = more sediment settles |
| `theta` | 0.1 | 0.02–0.5 | Angle of repose — lower = flatter slopes after thermal smoothing |
| `n_frames` | 300 | 50–2000 | Number of simulation frames |
| `grid_div` | 2 | 1–4 | Coarse grid factor — higher = faster but blockier |
| `dt` | 1.0 | 0.1–5.0 | Timestep multiplier — higher = faster erosion |
| `uplift_rate` | 0.001 | 0.0001–0.01 | Tectonic uplift per frame (for tectonic mode) |
| `wave_amplitude` | 0.002 | 0–0.01 | Coastal wave erosion amplitude |
| `noise_amplitude` | 0.3 | 0.01–1.0 | Initial topographic relief amplitude |
| `render_water` | "true" | true/false | Show water channels as bright overlay on terrain |

## Physics Notes

The model couples three state variables on a coarse grid (default 256×192 at grid_div=2):

1. **Height field h** — the terrain elevation. Modified by erosion (removal), deposition (addition), thermal smoothing (diffusion), and uplift.
2. **Water volume w** — accumulated rainfall, routed downhill to the lowest of 4 neighbors (steepest-descent). Partially evaporates each frame.
3. **Sediment load s** — eroded material carried by water. Erosion follows a stream power law: `Δs = K_e · w · |∇h| · dt`. Deposition occurs when sediment exceeds transport capacity: `capacity = K_d · w`.

The steepest-descent routing produces realistic dendritic drainage patterns. Dense network → fewer, deeper channels as the terrain matures.

**Initial conditions:** Multi-scale Perlin-like noise (4 octaves) for realistic initial topography. The terrain then evolves through drainage network development.

**Rendering:** Hillshade (315° azimuth, 45° altitude) combined with elevation. Water channels overlaid as bright highlights for the `render_water=true` case. Percentile contrast stretch maps [2%, 98%] to [0, 255].

## Outputs

- **IMAGE** — grayscale terrain with hillshade + optional water overlay
- **FIELD** — final height field as float32 (W×H)
- **SCALARS:** max_erosion, total_sediment, drainage_density

## Example Invocations

```bash
# Basic hydraulic erosion
python -m image_pipeline.pipeline --animate 137 --params '{"anim_mode":"hydraulic"}' --anim-duration 12

# Tectonic uplift with deep canyon incision
python -m image_pipeline.pipeline --animate 137 --params '{"anim_mode":"tectonic","uplift_rate":0.003,"n_frames":600}' --anim-duration 24

# Thermal smoothing only — gentle landscape
python -m image_pipeline.pipeline --animate 137 --params '{"anim_mode":"thermal","theta":0.08}' --anim-duration 10

# Coastal with wave erosion
python -m image_pipeline.pipeline --animate 137 --params '{"anim_mode":"coastal","wave_amplitude":0.005}' --anim-duration 12

# Combined — fastest landscape evolution
python -m image_pipeline.pipeline --animate 137 --params '{"anim_mode":"combined","rain_rate":0.02,"K_e":0.1}' --anim-duration 15
```
