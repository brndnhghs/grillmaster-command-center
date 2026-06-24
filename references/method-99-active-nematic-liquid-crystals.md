# Method 99 — Active Nematic Liquid Crystals

**Author:** Research Survey — soft matter physics / active nematodynamics

**Algorithm:** 2D Q-tensor continuum model of active nematic liquid crystals. Evolves the traceless symmetric Q-tensor field (Qxx, Qxy) on a 256×171 grid with explicit Euler integration. Glyph-based render: short line segments at each cell show the local director orientation. Color = director angle (periodic hue), brightness = order parameter S.

**Parameters:**

| Param | Range | Default | Effect |
|-------|-------|---------|--------|
| `activity` | −0.2–0.2 | 0.12 | Extensile >0 = defects self-propel |
| `elastic_d` | 0.01–2.0 | 0.2 | Orientational stiffness |
| `A_landau` | −0.5–0.1 | −0.2 | Ordering depth (S_eq ≈ 0.63) |
| `noise_amp` | 0–0.15 | 0.05 | Defect nucleation trigger |
| `substeps` | 1–50 | 15 | PDE steps per frame (higher = faster) |

**Animation Modes:** evolve, activity_sweep, quench, shear, defect_garden, contractile

**Performance:** ~8.5s static, ~16s for 120-frame animation. Architecture A.

**Dynamics:** ~4.5°/frame director rotation, 543° total over 120 frames.

**Invocation:**
```bash
python3 -m image_pipeline.pipeline --animate 99 --seed 42069 --params '{"anim_mode":"evolve","n_frames":150}'
```
