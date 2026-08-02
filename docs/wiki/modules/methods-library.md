# Module: `methods/` (Method Library)

## Purpose
The generative heart of the Image Pipeline — **542 registered methods** across 16 category sub-packages under `methods/`. Each method is a self-contained generator (fractal, simulation, filter, pattern, shader, ML model, …) that registers itself via the `@method` decorator. The library is auto-discovered: dropping a new `*.py` into a category package is enough to make it appear in the UI and the registry.

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

## Category Breakdown (542 methods)

| Category | Methods | What lives here |
|----------|---------|----------------|
| `gpu_shaders` | 159 | GPU shader nodes (moderngl GLSL 330 + webgl2 parity, typed uniforms) |
| `simulations` | 113 | Internal-loop sims (Gray-Scott, Boids, DLA, fracture) — Architecture A |
| `filters` | 86 | Post-style image transforms (see [`core-postprocess.md`](core-postprocess.md) for the filter layer) |
| `patterns` | 69 | Tiling, weaving, generative pattern art |
| `math_art` | 28 | Math-driven generative visuals |
| `channels` | 21 | CHOP-style channel nodes (LFO, Counter, Beats, Envelope, Math, Logic, Strobe, Burst, AgeHeat) |
| `fractals` | 16 | Escape-time & orbit-trap fractals (Mandelbrot, Julia, …) |
| `codegen` | 13 | Code-generation / shader-source methods |
| `compositing` | 10 | Blend / composite nodes (see [`core-compositing.md`](core-compositing.md)) |
| `cli_tools` | 8 | CLI-wrapper methods (pyfiglet ASCII art, etc.) |
| `io` | 7 | Input/output nodes (webcam, file) |
| `ml_models` | 7 | ML methods (Stable Diffusion, CLIP, ComfyUI bridge) — lazy torch |
| `analysis` | 1 | Analysis nodes |
| `client_3d` | 1 | Three.js client-3D nodes (rendered by the sidecar) |
| `p5_sketches` | 1 | p5.js sketch nodes |
| `system` | 2 | System / control nodes (e.g. `__timeline__`) |

> **Count note:** the registry is authoritative — `len(registry.get_all())` at import time. Category membership moves as methods are renumbered; `tools/next_id.py` allocates IDs and the migration tooling in `tools/` keeps tests in sync.

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
[`image_pipeline/methods/__init__.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/f689773c452e24fa1bf1bbcf3e6817fb5304c81d/image_pipeline/methods/__init__.py) · [example: `fractals/fractal.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/f689773c452e24fa1bf1bbcf3e6817fb5304c81d/image_pipeline/methods/fractals/fractal.py)
