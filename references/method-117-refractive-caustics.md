# Method #117 — Refractive Caustics

**ID:** 117  
**Name:** Refractive Caustics  
**Category:** simulations  
**Tags:** `physics`, `animation`, `optics`, `water`, `simulation`, `caustics`  
**Time-out:** 120 s  
**Architecture:** B (per-frame re-render)

---

## Description

Animated light caustics formed by simulating sunlight passing through a
time-varying wavy water surface.  Uses the **caustic mapping** approach:

1. A heightfield `h(x, y, t)` is built as a sum of 3–12 sinusoidal wave
   components with random amplitudes, directions, frequencies, and phases.

2. The refraction displacement at each point is proportional to the water
   surface slope:  `dx = D · ∂h/∂x`, `dy = D · ∂h/∂y`.

3. The Jacobian determinant of the mapping `(x, y) → (x+dx, y+dy)` measures
   light concentration — where `det(J)` is small, light focuses to a bright
   caustic web.

4. **Chromatic aberration:**  R, G, B use slightly different `D` values
   (`D_R = D·(1+chromatic)`, `D_G = D`, `D_B = D·(1−chromatic)`), producing
   colour fringing at caustic edges.  Each channel's caustic intensity is
   mapped through a colour palette and the relevant component extracted.

Result: bright golden, teal, warm-white, fire-like, or plasma-coloured
web-like caustic patterns on a dark background, with rainbow fringing at
the sharpest edges.

---

## Physics

### Heightfield

```
h(x, y, t) = Σ_i  A_i · sin(kx_i·x + ky_i·y − ω_i·t + φ_i)
```

where for each wave `i`:
- `A_i` = amplitude (random, scaled by `wave_amplitude`)
- `kx_i = k_i·cos(θ_i)`, `ky_i = k_i·sin(θ_i)` (direction)
- `k_i ∈ [0.02, 0.08]` (wave number)
- `ω_i ∈ [0.5, 2.0]` (angular frequency)
- `φ_i ∈ [0, 2π)` (phase)

### Caustic intensity

First derivatives via `np.gradient`:
```
dhdx = ∂h/∂x     dhdy = ∂h/∂y
```

Second derivatives (gradient of gradient):
```
Hxx = ∂²h/∂x²    Hyy = ∂²h/∂y²    Hxy = ∂²h/∂x∂y
```

Jacobian determinant:
```
det(J) = (1 + D·Hxx) · (1 + D·Hyy) − (D·Hxy)²
```

Brightness:
```
brightness = 1 / |det(J)|   (clipped at 1e-10)
```

Each channel uses a different `D` (depth per channel), giving three
slightly different caustic maps.

### Post-processing

1. Log compression: `log(1 + brightness)`
2. Min–max normalisation to `[0, 1]`
3. Gamma correction: `intensity^0.7`
4. Palette mapping (smooth interpolated ramp) → RGB for each channel
5. Channel extraction: `R ← R_pal(R_depth)`, `G ← G_pal(G_depth)`,
   `B ← B_pal(B_depth)`
6. Gaussian blur `σ = 0.6` for natural glow

---

## Parameters

| Parameter          | Type  | Range     | Default    | Description                                      |
|--------------------|-------|-----------|------------|--------------------------------------------------|
| `wave_amplitude`   | float | 0.5–5.0   | 2.0        | Overall wave strength                            |
| `n_waves`          | int   | 3–12      | 6          | Number of superimposed wave components           |
| `depth`            | float | 0.5–3.0   | 1.5        | Water depth — controls caustic sharpness         |
| `chromatic_strength`| float| 0.0–0.5   | 0.3        | Chromatic aberration amount                     |
| `palette`          | str   | —         | `gold_teal`| Colour palette name                              |
| `time`             | float | 0.0–6.28  | 0.0        | Animation time (swept by animator)               |
| `anim_mode`        | str   | —         | `wave_train`| Animation mode                                   |
| `anim_speed`       | float | 0.1–3.0   | 1.0        | Speed multiplier                                 |

### Palette choices

| Palette      | Description                                           |
|--------------|-------------------------------------------------------|
| `gold_teal`  | Dark teal → turquoise → gold → bright yellow (default)|
| `warm_white` | Deep navy → blue → cool white → pure white            |
| `ocean`      | Black → dark blue → cyan → pale blue/white           |
| `fire`       | Black → deep red → orange → yellow → pale gold       |
| `plasma`     | Dark purple → magenta → hot pink → orange → pale     |

---

## Animation modes

| Mode               | Behaviour                                                    |
|--------------------|--------------------------------------------------------------|
| `wave_train`       | Standard traveling waves — waves propagate with time         |
| `chromatic_sweep`  | Chromatic aberration pulses sinusoidally; colours breathe    |
| `amplitude_swell`  | Wave amplitude swells in and out like breathing              |
| `multi_source`     | Two independent wave systems interact at different angles    |
| `rotation`         | Wave gradient field rotates slowly, sweeping caustics around |

All modes use smooth `sin`/`cos` modulation — no `abs(sin())`.

---

## Usage

### Single frame

```bash
python -m image_pipeline generate 117 --params '{"wave_amplitude": 2.0, "depth": 1.5}'
```

### Animation (Architecture B)

```bash
python -m image_pipeline animate 117 \
  --params '{"wave_amplitude": 2.0, "depth": 1.5, "anim_mode": "wave_train", "anim_speed": 1.0}' \
  --nframes 120 --fps 24
```

The animator sweeps `"time"` from 0 → 2π across the frame count.

---

## Files

| File | Purpose |
|------|---------|
| `image_pipeline/methods/simulations/refractive_caustics.py` | Method implementation |
| `references/method-117-refractive-caustics.md` | This reference document |

---

## Implementation notes

- Pure NumPy — no SciPy dependency.
- Second derivatives via `np.gradient` of first-derivative fields.
- Wave parameters seeded from the loop index so they are deterministic
  across frames (waves don't randomly jump between frames).
- Palette lookup uses smooth linear interpolation, not nearest-neighbour.
- All animation modes produce visually distinct caustic behaviour.
- Architecture B: the method renders a single frame for the given `time`;
  the animator calls it repeatedly with a swept `time` parameter.
