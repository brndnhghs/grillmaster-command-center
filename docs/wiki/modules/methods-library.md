# Module: `methods/` (Method Library)

## Purpose
The generative heart of the Image Pipeline — **373 registered methods** across the `methods/` tree (8 category sub-packages + 37 top-level `methods/*.py` files). Each method is a self-contained generator (fractal, simulation, filter, pattern, shader, ML model, …) that registers itself via the `@method` decorator. The library is auto-discovered: dropping a new `*.py` into a category package is enough to make it appear in the UI and the registry.

## Registration System
`methods/__init__.py` walks the package with `pkgutil.iter_modules` and imports every submodule, which triggers each `@method(...)` decorator to populate the global registry (see [`core-registry.md`](core-registry.md)). No manual list of methods is maintained — the filesystem *is* the index.

### The `@method` contract
Every method file declares its metadata at decoration time:
```python
@method(
    id="33",
    name="Fractal Explorer",
    category="fractals",
    tags=["classic", "fast", "animated", "expanded"],
    inputs={"image_in": "IMAGE"},
    params={
        "formula":   {"default": "mandelbrot",
                      "choices": ["mandelbrot", "julia", "burning_ship", ...]},
        "iterations": {"min": 50, "max": 2000, "default": 200},
        "zoom":       {"min": 0.5, "max": 100000.0, "default": 1.0},
        "colormap":   {"default": "none"},
        "smooth":     {"default": True},
        # ...13 params total on Fractal Explorer
    },
)
def method_fractal(out_dir: Path, seed: int, params=None):
    ...
```
Key fields (full list in [`core-registry.md`](core-registry.md)): `id`, `name`, `category`, `tags`, `params` (default + optional `min`/`max`/`choices`/`description`), `inputs`/`outputs` (port-type strings), and `new_image_contract` (reads upstream image from in-memory `_input_image` instead of a disk path).

### How to add a method
1. Create `image_pipeline/methods/<category>/my_method.py`
2. Import `from ...core.registry import method` and `from ...core.utils import save, W, H, ...`
3. Decorate a `def my_method(out_dir, seed, params=None)` function with `@method(...)`
4. Return nothing (write via `save()`) or return an `ndarray` / dict
5. It auto-registers on next server start / hot-reload — no other file to edit

## Category Breakdown (373 methods)

| Category | Methods | What lives here |
|----------|---------|----------------|
| `simulations` | 112 | Internal-loop sims (Gray-Scott, Boids, DLA, fracture) — Architecture A |
| `filters` | 87 | Post-style image transforms (see [`core-postprocess.md`](core-postprocess.md) for the filter layer) |
| `patterns` | 69 | Tiling, weaving, generative pattern art |
| `math_art` | 27 | Math-driven generative visuals |
| `fractals` | 16 | Escape-time & orbit-trap fractals (Mandelbrot, Julia, …) |
| `compositing` | 10 | Blend / composite nodes (see [`core-compositing.md`](core-compositing.md)) |
| `codegen` | 14 | Code-generation / shader-source methods |
| `system` | 1 | System / control nodes (e.g. `__timeline__`) |
| `top-level` | 37 | Methods defined directly in `methods/*.py` (not in a sub-package) |

> **Count note:** the table sums to 336; the remaining 37 methods live in **top-level** files directly under `image_pipeline/methods/` (e.g. `io_nodes.py`, `custom_shader.py`, `gpu_shaders.py`, `cli_tools.py`, `references.py`, `simulations_cellular.py`) and are auto-discovered alongside the category packages.

## Architecture split
- **Architecture A** (simulation): methods with an `n_frames` param, `anim_mode`, or a `simulation`/`sim` tag cook a full frame list internally and are cached by the executor (see [`core-arch.md`](core-arch.md)).
- **Architecture B** (stateless): one call = one frame, driven by `time` / `_timeline` / `anim_mode`.

## Dependencies
- `core/registry.py` — the `@method` decorator and `MethodMeta`
- `core/utils.py` — `save`, `W`, `H`, palettes, dithering, sidecar protocol
- `core/animation.py` — `capture_frame` for Architecture-A sims
- Optional per-method: `cv2`, `matplotlib`, `torch`, `moderngl`, `pyfiglet`, `qrcode` (imported lazily so the server runs without them)

## Consumers
- `server.py` — serves `/api/methods`, `/api/node-defs`; executes methods via `GraphExecutor`
- `pipeline.py` — CLI batch execution via `registry.resolve_keys()`
- `core/graph.py` — builds `NodeDef`s from each method's metadata

## Source
[`image_pipeline/methods/__init__.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/image_pipeline/methods/__init__.py) · [example: `fractals/fractal.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/image_pipeline/methods/fractals/fractal.py)
