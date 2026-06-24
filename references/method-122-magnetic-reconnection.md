# Method #122 — Magnetic Reconnection

**ID:** 122  
**Name:** Magnetic Reconnection  
**Category:** simulations  
**Tags:** `physics`, `plasma`, `field-lines`, `magnetic`, `animation`  
**Time-out:** 120 s  
**Architecture:** B (per-frame re-render)

---

## Description

Glowing magnetic field lines of orbiting dipoles that stretch, pinch, and undergo
topological reconnection.  Each frame computes the vector potential of N point
magnetic dipoles (out-of-plane magnetization), differentiates to obtain the full
2D magnetic field, then traces streamlines through the field using Euler
integration with bilinear interpolation.

Field lines are coloured by their local **B-field angle** via an HSV palette,
producing smooth colour gradients that reveal the direction and topology of the
magnetic field.  Background glow radiates from dipole positions.  X-points
(nulls where |B| ≈ 0 and field topology reconnects) are highlighted with bright
cyan dots.

---

## Physics

### Vector potential

For each dipole `i` with:
- position `(pxᵢ, pyᵢ)`
- strength `mᵢ` (±1)
- orientation `θᵢ`

the vector potential at grid cell `(x, y)` is:

```
A_z(x,y) = Σᵢ  [ mᵢ · ((x−pxᵢ)·sin(θᵢ) − (y−pyᵢ)·cos(θᵢ)) / (|r−rᵢ|² + ε) ]
```

where `ε = 4.0` regularises the singularity at dipole positions.

### Magnetic field

Computed via centred finite differences (`np.gradient`):

```
Bx = ∂A_z/∂y
By = −∂A_z/∂x
|B| = sqrt(Bx² + By²)
φ_B = atan2(By, Bx)
```

### Streamline integration

1. Seed points are placed on a staggered grid across the canvas (spacing
   determined by `field_density`, default 24 px for density 3).
2. Each seed is integrated forward *and* backward using Euler steps of
   1.5 px, with bilinear interpolation of B at sub-pixel positions.
3. Integration stops when the streamline exits the canvas, `|B| < 0.001`
   (X-point / magnetic null), or 300 steps are exceeded.

---

## Animation modes

Five distinct modes, all producing **orbiting or oscillating dipole motion**:

| Mode | Dipoles | Behaviour |
|------|---------|-----------|
| `binary_orbit` | 2 (1+, 1−) | Classic reconnection — dipoles orbit each other; field lines stretch, pinch, and reconnect at each half-orbit. |
| `three_body` | 3 (+, −, +) | Chaotic exchange (Aref-like) — three dipoles in a rotating triangle; field topology cycles unpredictably. |
| `quadrupole` | 4 (2+, 2−) | Rotating square configuration — quadrupole field topology with four X-points. |
| `oscillating` | 2 (fixed positions) | Strengths oscillate sinusoidally — field lines breathe and reconnect periodically without dipole motion. |
| `driven` | 2 (pushed/pulled) | Separation oscillates slowly — dipoles are pushed together and pulled apart, producing periodic reconnection bursts. |

---

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `time` | float | 0.0 | [0, 2π] | Animation time (swept by the animator) |
| `anim_mode` | str | `binary_orbit` | — | One of the 5 animation modes |
| `anim_speed` | float | 1.0 | [0.2, 5.0] | Speed multiplier for dipole motion |
| `n_dipoles` | int | 2 | [2, 6] | Number of dipoles (used by some modes) |
| `strength` | float | 1.0 | [0.5, 3.0] | Dipole strength multiplier |
| `orbit_radius` | float | 120.0 | [50, 200] | Orbital separation in pixels |
| `field_density` | int | 3 | [1, 5] | Density of field lines; higher = more streamlines |
| `glow` | float | 0.5 | [0, 1] | Glow effect strength at dipole positions |
| `palette` | str | `cobalt` | — | Field line colour palette |

### Palettes

| Palette   | Description                                        |
|-----------|----------------------------------------------------|
| `cobalt`  | Dark blue → bright blue-white (default)            |
| `ember`   | Dark amber → bright warm gold                      |
| `aurora`  | Deep purple → vibrant green-blue                   |
| `phantom` | Purple → white (almost monochrome)                 |
| `frost`   | Ice blue → pure white                              |

Background field uses a two-tone blue↔gold polarity scheme — no rainbow hue cycling.

---

## Rendering pipeline

```
Two-tone background (blue↔gold from B_angle polarity)
  ↓
Dipole glow (blurred radial gradient at each dipole)
  ↓
Field streamlines (Euler integration, colorful from palette)
  ↓
Dipole markers (red circle +, blue circle -, white centre)
  ↓
X-point highlights (white dots at |B| ≈ 0)
  ↓
Draw X-point highlights (bright cyan dots at |B| nulls)
  ↓
Return (H, W, 3) uint8 numpy array
```

### Colour mapping

Each streamline segment is drawn with a single colour determined by the B-field
angle at its seed point:

```
t = (φ_B / 2π) mod 1     # normalise angle to [0, 1)
HSV = interpolate(palette[t])
RGB = HSV→RGB conversion
```

This is **physics-derived** — the colour reveals the local direction of the
magnetic field — NOT cosmetic cycling.

---

## Implementation notes

- `W = 768, H = 512` from `..core.utils`
- Pure numpy + PIL only — no scipy or skimage
- `seed_all(seed)` at function start, `seed_all(seed + int(t×100))` per-frame
- `save(arr, mn(118, "Magnetic Reconnection"), out_dir)` before return
- `capture_frame("118", arr)` for animation framework
- Bilinear interpolation used for B at sub-pixel streamline positions
- X-points detected as local minima of |B| where |B| < 0.1
