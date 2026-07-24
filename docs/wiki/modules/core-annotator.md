# Module: `core/annotator.py`

## Purpose
Demo / documentation overlay renderer. Stamps a method's identity and parameter ranges directly onto a generated image so a batch of outputs is self-describing (used by the `--demo` flag). Reads the `@method` metadata from the registry and renders it as a caption bar beneath the image.

## Responsibilities
- Overlay method id, name, tags, and per-parameter ranges onto an image
- Batch-annotate every method's default output in an output directory
- Pull display text straight from `MethodMeta` (no hard-coded labels)

## Key Functions

### `annotate_image(method_id, image_path, out_path=None) -> Path`
Opens `image_path`, looks up `get_meta(method_id)`, and builds an overlay:
- Title bar: `#<id>  <name>`
- Sub-line: `tags: <comma list>`
- One line per param: `description: <min>–<max> (default <d>)` when the param has `min`/`max`, else `description: default <d>`
- A dark caption strip is appended below the image; the method title and param lines are drawn with `PIL.ImageDraw`.

Raises `ValueError` if `method_id` is unknown. Returns the (possibly overwritten) output path.

### `annotate_batch(out_dir, method_ids, suffix="-demo") -> list[Path]`
For each `method_id`, finds its default output file via `meta.filename()`, annotates it to `<id>-<name-slug><suffix>.png`, and collects the resulting paths. Skips methods with no output file present.

## Overlay Layout
```
┌─────────────────────────────────┐
│  Image content (method output)  │
│                                 │
│  ── overlay bar ──             │
│  #07 Fractal (Mandelbrot)      │
│  tags: classic, fast            │
│  iterations: 50–300 (150)      │
│  viewport: [-2,1]×[-1.5,1.5] │
│  colormap: sin-loop            │
└─────────────────────────────────┘
```

## Dependencies
- `numpy`, `PIL` (`Image`, `ImageDraw`, `ImageFont`)
- `core/registry.py` — `get_meta`, `get_all`

## Consumers
- `pipeline.py` — `--demo` flag renders annotated outputs for every generated method
- Not wired into `server.py` (demo mode is CLI-only)

## Known Assumptions
- Uses `/System/Library/Fonts/Menlo.ttc` when present; falls back to `ImageFont.load_default()` on other platforms
- Caption height scales with the number of parameter lines (`20 + len(lines) * 18`)
- Output is always RGB (converts non-RGB inputs)

## Source
[`image_pipeline/core/annotator.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/image_pipeline/core/annotator.py)
