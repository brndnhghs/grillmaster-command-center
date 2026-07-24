# Module: `core/utils.py`

## Purpose
Shared utilities for all node methods — saving images, writing sidecar files, normalization, naming, canvas dimension management, color palettes, dithering, and image loading.

## Responsibilities
- Dynamic canvas dimension system (`W`, `H` resolve per-thread via ContextVar)
- Image saving with in-memory capture support (live mode avoids disk)
- Sidecar protocol: `write_scalars`, `write_field`, `write_particles`, `write_mask`
- Palette definitions and quantization (nearest-neighbor + linear interpolation)
- Ordered and Floyd-Steinberg dithering
- Image loading from disk (with canvas resize)
- Per-thread method ID context for node renumbering safety
- PIL, cv2, and numpy patches for dynamic dimension interop
- Font loading for text methods
- `seed_all()` for deterministic RNG setup

## Dynamic Canvas System

### `_DynDim` class
Proxy object that resolves canvas dimensions from a thread-local `ContextVar`. Every method file that does `from ...core.utils import W, H` gets these objects once at import time.

**Protocol support:**
- `__index__` → numpy `np.zeros((H, W, 3))`, `range(W)`
- `__int__` → PIL resize, cv2 operations
- `__float__` → arithmetic
- `__array__` → numpy interop (`np.mgrid[:H,:W]`)
- All arithmetic operators → return plain `int`, not `_DynDim`

### Functions
| Function | Description |
|----------|-------------|
| `set_canvas(w, h)` | Activate canvas for current thread, returns reset token |
| `reset_canvas(token)` | Restore previous canvas |
| `get_canvas()` | Return (width, height) |

## Sidecar Protocol

| Function | File written | Type |
|----------|-------------|------|
| `write_scalars(node_dir, **kwargs)` | `scalars.json` | Named float scalars |
| `write_field(node_dir, arr)` | `field.npy` | H×W float32 ndarray |
| `write_particles(node_dir, arr)` | `particles.npy` | N×4 float32 [x,y,vx,vy] |
| `write_mask(node_dir, arr)` | `mask.npy` | H×W float32 [0,1] |

All sidecar functions honour the per-thread sidecar context — in live mode, data is collected in memory instead of written to disk.

## Image I/O

| Function | Description |
|----------|-------------|
| `save(arr, name, out_dir)` | Save ndarray or PIL Image to PNG. Captures to in-memory sink when active; skips disk in live mode |
| `load_input(path, target_w, target_h)` | Load external image, resize to canvas, return float32 [0,1] |
| `wired_source_rgb(params, w, h)` | Get wired upstream image or None |

## Palettes and Dithering

### `PALETTES` dict
25+ named color palettes: `bw`, `grayscale`, `amber`, `green`, `gameboy`, `cga`, `pico8`, `nes`, `apple2`, `zxspectrum`, `c64`, `megadrive`, `sms`, `atari2600`, `amiga`, `warm`, `cool`, `vapor`, `sepia`

| Function | Description |
|----------|-------------|
| `quantize_to_palette(arr, palette_name)` | Nearest-neighbor quantization in RGB space |
| `apply_palette(arr, palette_name)` | Luminance-based palette mapping via linear interpolation |
| `ordered_dither(arr, levels, bayer)` | Bayer ordered dither |
| `floyd_steinberg_dither(arr, levels)` | Error diffusion dither |

## Patch System
Three monkey-patches installed at module load time:
1. **PIL patch**: wraps `Image.new()` and `Image.resize()` to call `operator.index()` on size tuples
2. **cv2 patch**: wraps `cv2.resize()`, `warpAffine()`, `warpPerspective()` to convert size tuples to ints
3. **numpy patch**: wraps `np.mgrid` and `np.ogrid.__getitem__()` to fix bounds resolution

## Dependencies
- `numpy`, PIL (`Image, ImageDraw, ImageFont, ImageFilter, ImageOps`)
- stdlib: `contextvars`, `math`, `operator`, `random`, `io`, `pathlib`, `threading`

## Consumers
Every method file imports from here. Also consumed by `core/animation.py`, `core/graph.py`, and `server.py`.