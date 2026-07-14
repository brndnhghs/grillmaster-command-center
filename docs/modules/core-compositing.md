# Module: `core/compositing.py`

## Purpose
53 blend modes + layout compositing utilities. Extracted from the monolithic legacy pipeline.

## Responsibilities
- Provide `blend_two()` with 53 blend modes
- Image loading, resizing, and normalization helpers
- Saving composited output

## Public Interfaces
| Function | Description |
|----------|-------------|
| `blend_two(a, b, mode)` | Blend two float32 RGB images [0,1] with named mode |
| `load_as_array(path)` | Load image as float32 [0,1] |
| `resize_to_target(img, tw, th)` | Resize to target dimensions |
| `save(arr, path)` | Save ndarray/PIL Image to path |

## Blend Modes (53 total)
- Normal, dissolve
- Darken: multiply, color-burn, linear-burn, darken-only, darker-color
- Lighten: screen, color-dodge, linear-dodge/addition, lighten-only, lighter-color
- Contrast: overlay, soft-light, hard-light, vivid-light, linear-light, pin-light, hard-mix
- Invert: difference, exclusion, subtract, divide
- Channel: hue, saturation, color, luminosity
- Legacy: blend, glow, reflect, freeze, heat, stamp, grain-merge, grain-extract, pn, phoenix
- Geometric: geometric-mean, average, negation, extremity

## Dependencies
- `cv2`, `numpy`, PIL
- `registry.py` — `get_meta` (for node-aware compositing)

## Consumers
- `methods/compositing/blend.py` — Blend node method
- `pipeline.py` — CLI compositing pipeline