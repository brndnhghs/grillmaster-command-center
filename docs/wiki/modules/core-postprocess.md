# Module: `core/postprocess.py`

## Purpose
OpenCV-backed post-processing filter library. Applies a single named effect (or a JSON spec with parameters) to a generated image and writes the result back. This is the "filter" layer that sits on top of method output — oil painting, edge detection, color grading, geometric warps, glitch, etc.

## Responsibilities
- Apply one effect to an image file via `apply_filter()`
- Apply the same effect across a batch of method outputs via `apply_filter_batch()`
- Parse string / dict filter specs (`--filter oil` or `--filter '{"effect":"colormap","colormap":"ocean"}'`)
- Named-color resolution for color-typed effects
- Lazy-load `cv2` so the module imports even when OpenCV is absent (effect raises at call time)

## Key Functions

### `apply_filter(image_path, filter_spec, out_path=None)`
Loads `image_path` as `uint8` RGB, parses `filter_spec`, runs the matching effect branch, and writes the result to `out_path` (defaults to overwriting `image_path`). Returns the output path.

### `apply_filter_batch(out_dir, method_ids, filter_spec, suffix="")`
Runs `apply_filter()` over each method's default output PNG in `out_dir`. Used by the CLI compositing pipeline.

### `_parse_spec(filter_spec)`
Normalizes a string (`"oil"`) or dict (`{"effect":"oil","radius":6}`) into a uniform spec dict.

### `_parse_color(name)`
Resolves a named color (`"red"`, `"teal"`, `"amber"`, …) to an `(R,G,B)` tuple; unknown names fall back to white.

### `_ensure_cv2()`
Lazy `import cv2` — keeps the module importable without OpenCV installed.

## Effect Catalog (~56 effects)
Effects are dispatched by `effect ==` string match. Categories:

- **Artistic:** `oil`, `detail`, `sharpen`, `emboss`, `pencil`, `cartoon`, `watercolor`, `sketch`, `neon`, `glow`, `rust`, `canvas`, `noise`, `frosted`
- **Edge / detection:** `edge` (with `mode` low/high), `halftone`, `mosaic`, `stained_glass`, `posterize`, `threshold`, `solarize`, `equalize`, `auto_contrast`, `desaturate`, `monochrome`, `invert`
- **Color grading:** `colormap`, `palette`, `chroma`, `vignette`, `bloom`, `pixelate`, `duotone`, `gradient_map`, `color_boost`, `channel_mix`, `cross_process`, `sepia`, `temperature`, `color_balance`, `split_tone`, `bleach_bypass`, `teal_orange`
- **Geometric warp:** `swirl`, `twist`, `ripple`, `bulge`, `pinch`, `kaleidoscope`, `waves`, `sphere`, `fisheye`, `cylinder`, `flag`, `lens`, `squeeze`
- **Glitch / stylize:** `glitch`, `morph`, `clahe`

Each effect reads its parameters from the spec dict with sensible defaults (e.g. `oil` → `radius=4, levels=10`; `bloom` → `strength=0.5`).

## Dependencies
- `cv2` (lazy import), `numpy`, `PIL`
- No internal pipeline imports — pure image-in / image-out

## Consumers
- `server.py` — imports `apply_filter` (filter layer on generated output)
- `pipeline.py` — drives filters via the `--filter` CLI flag (`--filter oil`, `--filter '{"effect":"colormap","colormap":"ocean"}'`)

## Known Assumptions
- Input is always loaded as `uint8` RGB; float32 `[0,1]` arrays are NOT accepted (use `core/utils.save` first)
- Spec parsing is permissive: an unknown `effect` string silently passes the image through unchanged
- `oil` prefers `cv2.xphoto.oilPainting`; falls back to bilateral filter when the xphoto module is unavailable

## Source
[`image_pipeline/core/postprocess.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/image_pipeline/core/postprocess.py)
