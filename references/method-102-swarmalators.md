# Method 102 — Swarmalators

**Author:** Research Survey — sync + swarm systems (O'Keeffe, Hong & Strogatz, 2017)

**Algorithm:** N agents with spatial position (x,y) AND oscillator phase θ ∈ [0, 2π). Phase affects spatial attraction ("like attracts like") and proximity gates synchronization. RK4 integration with O(N²) pairwise interactions via numpy broadcasting.

**Parameters:**

| Param | Range | Default | Effect |
|-------|-------|---------|--------|
| `n_agents` | 50–2000 | 400 | Number of swarmalator agents |
| `J_attract` | −2–3 | 0.5 | Like-phase attraction (>0 = same phase clusters) |
| `K_sync` | −3–3 | 2.0 | Phase coupling strength |
| `freq_spread` | 0.1–3 | 1.5 | Natural frequency diversity |
| `self_prop` | 0–3 | 1.2 | Self-propulsion (0 = static ring) |
| `repulsion` | 0.2–5 | 1.5 | Short-range repulsion |

**Animation Modes:** evolve, param_sweep, frequency_gradient, external_drive, multi_species, quenched

**Performance:** ~12s static, ~60s for 120-frame animation. Architecture A.

**Dynamics:** 23.6°/frame phase rotation, 0.36 px/frame spatial drift.

**Visual:** Glowing colored dots on dark background. 5×5 glow kernel. Hue = phase angle.

**Invocation:**
```bash
python3 -m image_pipeline.pipeline --animate 102 --seed 42069 --params '{"anim_mode":"evolve","n_frames":150}'
```
