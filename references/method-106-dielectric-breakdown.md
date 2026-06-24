# Method 106 — Dielectric Breakdown (Lichtenberg Figures)

**Author:** Research Survey — Laplacian growth / electrical discharge
**Algorithm:** Stochastic fractal branching driven by a Laplace potential field. Branch tips with steeper gradient `|∇φ|` are more likely to extend, with probability P ∝ `|∇φ|^η / dielectric_strength`.

**Physics extensions:**
- **Per-cell temperature**: each cell born hot, cools every frame — white → orange → red → purple → dark
- **Thermal mass**: trunk/junction cells (more neighbors) retain heat longer than isolated tips
- **Dielectric variation**: multi-octave sinusoidal noise field creates preferential growth paths
- **Micro-arc sparks**: when a tip grows within 1 cell of another branch, a bright flash bridges the gap
- **Multi-seed competition**: separate trees grow simultaneously; guaranteed sparks on contact

**Parameters:**

| Param | Range | Default | Effect |
|-------|-------|---------|--------|
| `eta` | 0.1–3.0 | 1.2 | Branching exponent: low → dense bushes, high → sparse straight |
| `growth_rate` | 1–30 | 8 | New cells added per frame |
| `cool_rate` | 0.85–0.999 | 0.976 | Temperature decay per frame (calibrated to seed lifespan) |
| `dielectric` | 0–1 | 0 | Material variation strength (0 = uniform, 1 = strong variation) |
| `spark_prob` | 0–1 | 0 | Probability of micro-arc when tip nears another branch |
| `seeds` | 1–5 | 1 | Number of seed points |

**Animation Modes:** grow, directional, strike_and_decay, multi_seed

**Performance:** ~2s static, ~90s for 500-frame animation. Architecture A. Coarse grid (192×128) → 4× BILINEAR upscale.

**Visual:** White-hot fractal branching on dark background, cooling through orange → red → purple. Temperature-dependent glow widens around hot cells. Sparks flash white.

**Invocation (maxed out):**
```bash
python3 -m image_pipeline.pipeline --animate 106 --seed 42069 \
  --anim-duration 21 --anim-fps 24 \
  --params '{"anim_mode":"multi_seed","n_frames":500,"eta":1.0,"growth_rate":14,"cool_rate":0.980,"dielectric":0.6,"spark_prob":0.35}'
```
