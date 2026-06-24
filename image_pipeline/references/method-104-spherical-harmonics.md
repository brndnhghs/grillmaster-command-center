# Method 104 — Animated Spherical Harmonics

**Research domain:** Quantum mechanics / mathematical visualization
**Author:** Research-to-Method pipeline / Survey sub-agent

## Description

Renders glowing 3D isosurfaces of spherical harmonics Y_l^m(θ,φ) — the angular part of hydrogen-like atomic orbitals. Sweeps through quantum numbers l (0→max) and m (-l→l) with a slowly orbiting camera, producing morphing lobe structures in electric blue (positive phase) and fiery orange (negative phase).

From the simple spherical s-orbital (l=0) through dumbbell p-orbitals (l=1) and cloverleaf d-orbitals (l=2) to intricate multi-lobed f/g-orbitals (l=3+), each shape smoothly morphs into the next against a deep space background. The result resembles a $10K scientific visualization rendered in Blender — produced from a single Python method.

## Algorithm

- **Surface generation:** Evaluate Y_l^m(θ,φ) on a 100×140 grid of (θ, φ) using scipy.special.sph_harm_y. Separate into positive (Re > 0) and negative (Re < 0) phase lobes. Convert to 3D Cartesian points where r(θ,φ) = |Re(Y_l^m)| × amplitude.
- **Camera:** Orthographic projection with slow orbit (Y-axis rotation + slight X-axis tilt).
- **Rendering:** Depth-sorted point cloud rendered as colored circles with brightness proportional to spherical harmonic magnitude. Gaussian blur (radius=2) creates glow.
- **Animation sweep:** Cycles through l=0→max_l, for each l iterating m=-l→l. Each frame switches to a new (l, m) combination with a slightly different camera angle.

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| max_l | 1–8 | 5 | Maximum angular momentum quantum number |
| amplitude | 0.5–3.0 | 1.5 | Orbital size scale |
| glow_strength | 0.5–3.0 | 1.5 | Glow intensity multiplier |
| n_frames | 60–400 | 180 | Number of simulation frames |

## Animation Modes

| Mode | Description | Visual effect |
|------|-------------|---------------|
| none | Static output | Final orbital in the sequence |
| evolve | Internal loop | Orbital morphing through l,m + camera orbit |

**Architecture A** — no `"time"` param. Use:
```
--animate 104 --params '{"anim_mode":"evolve","n_frames":180,"max_l":5}'
```

## Visual Characteristics

- Deep navy background with glowing electric blue (positve lobe) and fiery orange (negative lobe)
- Orbits morph smoothly from simple spheres through dumbbells, cloverleaves, and intricate multi-petal shapes
- Slow camera orbit reveals 3D structure
- Output resolution: 768×512
- Speed: fast (~4s for 180 frames)

## Example Invocation

```bash
# Static
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --methods 104 --force --no-cache

# Animation
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 104 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":180,"max_l":5,"glow_strength":1.5}'

# Higher resolution orbitals (more l, more frames)
PYTHONPATH=/Users/admin/Documents/GitHub/grillmaster-command-center \
python3 -m image_pipeline.pipeline --animate 104 --force --no-cache \
  --params '{"anim_mode":"evolve","n_frames":300,"max_l":7,"glow_strength":1.5}'
```

## Notes

- **Architecture A only.** Do NOT include `"time"` in params.
- Requires scipy (`sph_harm_y` from scipy.special).
- Surface sampling resolution is 100×140 for speed (~14K points per orbital). Increase N_THETA/N_PHI in the source for finer detail at the cost of render time.
- The l=0 (s-orbital) is a perfect sphere with only positive phase (no orange).
- Higher l values produce more lobes with finer detail — still visible at 768×512.
- MP4 output typically ~100-200 KB (well under Discord's 10 MB limit).
- Nothing like this exists in the current catalog — it's the first quantum/mathematical 3D visualization.
