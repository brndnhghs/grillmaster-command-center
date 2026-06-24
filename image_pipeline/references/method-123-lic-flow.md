# Method 123 — Animated LIC Flow

**Category:** simulations  
**File:** `simulations/lic_flow.py`  
**Author:** Automated Survey (LIC / Flow Visualization)

## Description

Line Integral Convolution (LIC) maps a white noise texture through a time-evolving vector field by convolving noise values along streamlines. Every pixel gets a correlated value that reveals the full flow topology — sinks, sources, vortices, and saddle points appear as distinct textural features. Unlike particle advection (which leaves empty space between tracers), LIC fills the entire canvas with dense, silk-like flow texture.

## Key Parameters

| Param | Range | Default | Description |
|-------|-------|---------|-------------|
| `anim_mode` | choices | `double_gyre` | Flow topology: `dipole`, `vortex_pair`, `double_gyre`, `von_karman`, `saddle`, `spiral` |
| `color_mode` | choices | `direction` | `direction` (two-tone cool/warm by angle), `magnitude` (speed heatmap), `phase` (full hue desaturated), `thermal` (black→red→orange→white), `bipolar` (curl: warm=CW, cool=CCW) |
| `conv_length` | 10–80 | 30 | Streamline integration steps — longer = more smeared texture |
| `ds` | 0.3–3.0 | 1.0 | Streamline step size in pixels |
| `flow_scale` | 0.1–3.0 | 1.0 | Overall flow magnitude multiplier |
| `noise_res` | 1–4 | 2 | Noise resolution divisor (2 = 384×256 compute, upscaled to 768×512) |
| `blur_radius` | 0–4.0 | 0.5 | Gaussian blur on output for smoothness |
| `advection` | 0–10 | 2.0 | Noise advection strength. **0=static texture** (only morphs with field evolution). Higher = texture flows faster along streamlines. This is what creates visible movement *in the direction of flow* — without it, the LIC texture only changes because the field itself evolves, not because the texture advects. |

## Animation Modes

All 6 anim_modes are true animation — the vector field evolves with `t`, producing smoothly morphing flow textures:

- **dipole** — source-sink pair orbiting the canvas center. Tangled streamlines with fast flow near the singularities.
- **vortex_pair** — two counter-rotating vortices slowly drifting apart. Beautiful braided flow texture.
- **double_gyre** — classic periodic gyre with transport between gyres. LIC benchmark topology.
- **von_karman** — vortex street shedding behind a bluff body. Alternating vortices drift downstream.
- **saddle** — hyperbolic stagnation point, slowly rotating. Simple but elegant flow topology.
- **spiral** — spiral sink with drifting center. Tangential flow wraps into the singularity.

## Visual Characteristics

- **Dense silk-like texture** fills 100% of the canvas in all modes
- **30 distinct looks** (6 modes × 5 color modes)
- Flow direction mapped to two-tone cool/warm palette (default) avoids rainbow
- Thermal and bipolar color modes derive from physics (speed and vorticity)
- Sub-pixel streamline integration gives anti-aliased texture
- Typical per-frame PNG size: 250–460 KB (dense texture, compresses well)
- First-to-last Δ (double_gyre): ~0.14 — clear evolution across full cycle

## Performance

- ~8–12 seconds per frame at `noise_res=2` (384×256 compute, 768×512 output)
- Use `noise_res=4` (192×128) for faster iteration, `noise_res=1` for full-res detail

## Example Commands

```bash
# Static render
python -m image_pipeline.pipeline --methods 123 --seed 69 --params '{"anim_mode":"double_gyre","color_mode":"direction","conv_length":30}'

# Animation (Architecture B — time-based per-frame re-call, with noise advection)
python -m image_pipeline.pipeline --animate 123 --params '{"time":0,"anim_mode":"double_gyre","color_mode":"direction","conv_length":20,"advection":3}' --anim-duration 4
```
