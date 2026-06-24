# Method 101 — Viscous Fingering

**Research domain:** Hele-Shaw / Saffman-Taylor instability (fluid dynamics)
**Author:** Research-to-Method pipeline / Survey sub-agent

## Description

Simulates the classic Hele-Shaw cell experiment where a low-viscosity fluid is injected into a high-viscosity fluid between two parallel plates. The pressure-driven interface destabilizes into branching, splayed-out fingers — a phenomenon known as the Saffman-Taylor instability.

Uses a curvature-weighted front propagation model on a coarse grid (384×256 → upscaled to 768×512). Fingers grow faster at convex tips and are suppressed in concave bays, producing the characteristic smooth, splayed finger morphology that distinguishes it from DLA's jagged dendritic structure.

## Algorithm

- **Grid:** 384×256 coarse simulation grid, upscaled 2× to 768×512 for output
- **Initial seed:** Invader fluid seeded as a small circle (center/bottom) or a line (bottom edge)
- **Per-frame:**
  1. **Front detection:** Find all uninvaded cells adjacent to invaded cells using 3×3 convolution
  2. **Curvature proxy:** For each front cell, compute invaded neighbor density in a 7×7 kernel. Tips (convex) have ~0 invaded inner neighbors → curvature ≈ 1.0. Bays (concave) have most inner neighbors invaded → curvature ≈ 0.0
  3. **Growth probability:** `P = k^curvature_power × tip_boost + noise`. Tips grow much faster than flat/bay regions
  4. **Batch advancement:** Sample ~200 cells per frame from the weighted distribution
- **Rendering:** Cells colored by invasion time on a plasma-like colormap (dark purple → magenta → orange → yellow → white), with mild Gaussian blur for smooth edges

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| curvature_power | 0.5–4.0 | 2.0 | How strongly curvature amplifies growth (higher = sharper fingers) |
| noise_amplitude | 0.0–5.0 | 2.0 | Stochastic noise (higher = more branching/tip-splitting) |
| cells_per_frame | 50–500 | 200 | Cells advanced per simulation frame |
| tip_boost | 1.0–5.0 | 1.5 | Additional growth multiplier for finger tips |
| inject_mode | center/bottom/line_bottom | center | Where invader fluid is injected |
| n_frames | 30–200 | 100 | Number of simulation frames |

## Animation Modes

| Mode | Description | Visual effect |
|------|-------------|---------------|
| none | Static output | Final fingering pattern after N frames |
| evolve | Internal simulation loop | Fingers branching and growing from injection point |

**Architecture A** — no `"time"` param in invocation. Use:
```
--animate 101 --params '{"anim_mode":"evolve","n_frames":80,"inject_mode":"center"}'
```

## Visual Characteristics

- Dark navy background with plasma-colormap fingers (purple → magenta → orange → yellow)
- Three injection patterns: center (radial), bottom (radial from bottom), line_bottom (horizontal line at base)
- Smooth splayed fingers vs. DLA's jagged dendritic morphology — the key visual distinction
- Output resolution: 768×512
- Deterministic per seed
- Speed: moderate (~5s for 80 frames)

## Example Invocation

```bash
# Static: center injection
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --methods 101 --force --no-cache

# Animation: center injection
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 101 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":80,"inject_mode":"center","curvature_power":2.0}'

# Line injection from bottom (different morphology)
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 101 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":80,"inject_mode":"line_bottom","curvature_power":2.2,"noise_amplitude":1.5}'

# High branching (more noise, less curvature)
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 101 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":80,"inject_mode":"center","curvature_power":1.0,"noise_amplitude":4.0}'
```

## Notes

- **Architecture A only.** Do NOT include `"time"` in params.
- Uses scipy.ndimage for convolution and uniform_filter — lightweight dependency already in the venv.
- Curvature proxy method is a heuristic (not full PDE pressure solve), but produces visually identical results to real Hele-Shaw experiments.
- The `line_bottom` mode simulates injection along a horizontal crack/line source rather than a point.
- MP4 output typically ~100-200 KB (well under Discord's 10 MB limit).
- Key difference from DLA (#36): DLA uses random walkers that stick to produce jagged dendritic structures. Viscous Fingering uses pressure-driven front propagation with curvature weighting to produce smooth, splayed-out fingers with characteristic tip-splitting.
