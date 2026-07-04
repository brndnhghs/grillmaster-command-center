# Grillmaster Command Center — Agent Guide

This document is for any agent creating or modifying node methods in this pipeline. Read it fully before touching any file.

---

## What we are building and why

Grillmaster Command Center is a **node-based generative image/video editor designed to be readable and extensible by LLM agents**. It takes the best of Houdini (typed named-attribute payloads flowing through wires) and TouchDesigner (a live, continuously-cooking graph with CHOP-style channel nodes), and is built so that an agent can read a single method file and immediately understand what it does, what it needs, and what it produces. All in-app LLM features (Node Doctor, node-tester batch fixes) run through the **Hermes agent — the sole LLM backend for all LLM calls**.

The pipeline currently generates 2D images. The architecture is deliberately designed to generalize to 3D (geometry, volumes, particle systems) in the future, so avoid hardcoding assumptions that are specific to 2D where possible.

**Why this architecture matters:** In Houdini, every node is an operation on a named attribute bus — data flows through wires, nodes transform it, and downstream nodes pick up what they need by name. We replicate that pattern here. Nodes produce typed outputs (IMAGE, SCALAR, FIELD, PARTICLES, MASK). Those outputs flow through wires to downstream nodes. A downstream node reads what it needs by name. No hidden state. No post-hoc magic. What a method writes is exactly what flows downstream.

The goal is **mutual legibility**: if you write a method file, a human reading it should understand exactly what it does. If a human reads a method file, you should be able to understand it and extend it. Every design decision in this repo is made with that contract in mind.

---

## Repository structure

```
image_pipeline/
  methods/          ← all node methods, one file per node
    simulations/    ← physics/biology sims (reaction-diffusion, boids, etc.)
    fractals/       ← escape-time and IFS fractals
    patterns/       ← geometric patterns, mazes, Voronoi, etc.
    math_art/       ← mathematical visualizations
    codegen/        ← programmatic/algorithmic generators
    filters/        ← image filters (applied to upstream image input)
    compositing/    ← blend, math merge, field combine, particle merge, apply mask
    cli_tools/      ← methods wrapping ffmpeg, ImageMagick, etc.
  core/
    graph.py        ← executor: topological sort, dirty flags, payload propagation
    registry.py     ← @method decorator and MethodMeta dataclass
    utils.py        ← save_image, write_scalars, write_field, write_particles, write_mask, apply_palette
    animation.py    ← capture_frame helper
  server.py         ← FastAPI app: /api/node-defs, /api/run-node, /api/graph/run, SSE streaming
ui/
  index.html        ← entire frontend (single file): node graph, wiring, UI
DESIGN.md           ← authoritative architecture document
AGENT_GUIDE.md      ← this file
```

---

## How execution works

1. The user builds a graph in the browser: nodes connected by typed wires.
2. On run, the frontend POSTs the graph to `/api/graph/run`.
3. The executor (`graph.py`) topologically sorts the nodes (Kahn's algorithm).
4. If any node has `render=True`, execution stops at that node — it is the terminal.
5. Nodes with `dirty=False` and no dirty upstream skip execution and use cached output.
6. Each node runs its method function. The method writes files to `node_dir/`.
7. After each node, the executor reads its outputs and stores them in `flat_outputs[node_id]`.
8. Upstream scalar values propagate downstream automatically (named attribute inheritance). Explicit wires take priority.
9. The final node's image is streamed back to the frontend via SSE.

**Image input routing:** The executor writes the upstream node's image to `node_dir/_input.png` and injects its path into `run_params["input_image"]`. If a method wants to use the upstream image, it reads from `params.get("input_image", "")`. There is **no hidden compositing** — whatever the method writes to its output PNG is exactly what flows downstream.

### Frame and animation

Methods receive `params['frame']` (int, default 0) and `params['frame_seed']` (int) on every execution. Use these to vary output over time:

```python
frame = params.get('frame', 0)
# frame_seed already incorporates frame into the node seed — use it for np.random:
np.random.seed(params.get('frame_seed', seed))

# For smooth animation, map frame to a continuous parameter:
t = frame / max(params.get('total_frames', 48) - 1, 1)  # 0.0 → 1.0
angle = t * 2 * math.pi  # full rotation over the sequence
```

Methods that don't read `frame` at all still animate naturally because their seed (`frame_seed`) changes each frame, so any stochastic draws produce different results.

To render a frame sequence use `POST /api/graph/render-sequence` — it saves `frame_NNNN.png` files to `output/sequences/<name>/` and streams SSE progress events. Individual frames are available at `GET /api/sequences/<name>/<frame>`.

#### Making a method animate in **live mode**

Live mode (the 📺 continuous cook loop) is the real-time target. Whether a method moves there depends entirely on what it reads:

- **Read one of `time`, `frame`, or `frame_seed`.** The live loop advances the timeline clock (`t`/`phase`, 0→1 over a 300-frame window) *and* injects a monotonic, unbounded `params['time'] = float(frame)`. A method that keys its state off any of these evolves continuously. A method that reads none of them renders the same image every frame and looks frozen — that is a method bug, not a live-mode bug.
- **Use `time` (unbounded) for open-ended evolution; use `t`/`phase` (0→1, clamped) for looping motion** like `sin(phase)`. Don't gate motion behind a large fixed floor that swallows small time deltas unless you intend a static single-frame preview.
- **Never require a prior "Run".** The client sends nodes as `dirty=False` after a Run; live mode force-dirties every node so it always re-cooks. Don't add state that assumes frames arrive in order or exactly once.

The live architecture and its four non-regression invariants are documented in `DESIGN.md` → "Live mode", and locked by `image_pipeline/tests/test_live_regression.py`. If you change how a method reads time, run that suite.

**Simulations have an extra contract.** If your node *accumulates state* over time (a CA grid, particles, an evolving field), it must keep its **last state and step it one step per frame** — the persistent stateful pattern, which runs forever at constant cost (#18 works this way) — or, for a finite scrubbable sequence, the cook-a-window Architecture-A pattern. Never a stateless node that re-simulates up to `time` each frame, which gets slower and slower as the timeline runs (the #18 bug). Before writing or rewriting any sim node, follow `docs/prompts/simulation-node-render-contract.md`.

#### The `-1.0` sentinel lesson (SCALAR override ports)

If a param is both a UI control *and* a wireable SCALAR override, do **not** use `-1.0` (or any in-range value) as a "not wired" sentinel checked with `is not None`. The client always sends the param at its default, so `params.get(name) is not None` is always true and the override permanently clobbers the UI value. Check the sentinel explicitly:

```python
sel = params.get("rule_select")
if sel is not None and float(sel) >= 0:   # a wired channel sends 0..1
    effective_rule = RULE_NAMES[int(float(sel) * len(RULE_NAMES)) % len(RULE_NAMES)]
else:                                       # -1.0 default → honour the UI param
    effective_rule = params.get("rule", "conway")
```

(This is exactly what silently broke method #18's `rule` / `seed_pattern` / `size` controls.)

---

## The method file contract

Every method lives at `image_pipeline/methods/{category}/filename.py`. It must satisfy the following contract exactly.

### 1. Register with `@method`

```python
from ..core.registry import method

@method(
    name="My Method Name",
    id=42,                          # unique integer, never reuse
    tags=["simulation", "pattern"], # used for search/filtering
    outputs={                       # declare EVERY output you write
        "image": "IMAGE",
        "luminance": "SCALAR",
        # add more if you write sidecars:
        # "field": "FIELD",
        # "r": "SCALAR",
        # "mask": "MASK",
    },
    # add inputs= if you have typed wire inputs:
    # inputs={"particles": "PARTICLES"},
)
def run(out_dir: Path, seed: int, params: dict) -> None:
    ...
```

Rules:
- `outputs=` is **required**. At minimum `{"image": "IMAGE", "luminance": "SCALAR"}`.
- Declare every sidecar you write (field, particles, mask, named scalars). Undeclared sidecars are ignored by the node graph, and the pre-commit audit (`tools/audit_methods.py`) fails on them.
- `id` must be unique across all methods — get it from `uv run python tools/next_id.py`, never pick or reuse one manually. The registry **raises at import time** on a duplicate id from a different module, which breaks server boot until fixed.
- Set `description=` — one sentence saying what the node does. It appears in the UI tooltip and in the Node Doctor's context; a nameless node is illegible to the next agent.
- Never use positional arguments on `@method`.

### 2. Always produce an image — on every code path

Preferred (return-dict contract): return `{"image": arr}` where `arr` is float32 (H, W, 3) in [0, 1]. Also accepted: returning an ndarray or PIL image, or (legacy) writing a PNG to `out_dir` and returning nothing — the executor reads the last non-`_`-prefixed PNG as the node's image.

What must never happen is a code path that **silently exits with no image**. Concretely, any broad `except Exception:` handler must do one of three things:

```python
except Exception as e:
    # 1. return an error image:
    return {"image": np.zeros((H, W, 3), dtype=np.float32)}
    # …or 2. save a fallback PNG:
    save(blank, "fallback.png", out_dir)
    # …or 3. re-raise — the executor paints a dark-red error placeholder
    #    and surfaces the traceback on the node in the UI:
    raise
```

Narrow compat handlers (`except (AttributeError, TypeError):` shims that fall through to the normal save/return) are fine. The pre-commit audit enforces exactly this rule.

### 3. Temp files use `_` prefix

Any intermediate file written during processing must start with `_`:

```python
tmp_path = out_dir / "_ffmpeg_input.png"
frame_path = out_dir / "_frame_0001.png"
```

The executor excludes `_`-prefixed files from output detection. If cleanup fails, the prefix ensures these files are never mistaken for output.

### 4. Explicit imports — nothing implicit

Never call a helper without importing it at the top of the file. Common imports:

```python
from ..core.utils import (
    save_image,
    apply_palette,
    write_scalars,
    write_field,
    write_particles,
    write_mask,
)
from ..core.animation import capture_frame
```

All dependencies must be in `requirements.txt`. Use `uv run` or `.venv/bin/python`, never system Python.

### 5. Handle `input_image` gracefully

If the method accepts an upstream image, read it from `params.get("input_image", "")`. Always guard with a truthiness check — the param is an empty string when no upstream is connected:

```python
input_path = params.get("input_image", "")
if input_path and Path(input_path).exists():
    img = np.array(Image.open(input_path)).astype(np.float32) / 255.0
else:
    img = np.zeros((height, width, 3), dtype=np.float32)  # sensible default
```

If your method explicitly accepts an image wire, declare it: `inputs={"image_in": "IMAGE"}` on `@method`.

---

## Sidecar protocol — exposing outputs beyond the image

The sidecar protocol is how methods surface computed values as typed wire outputs in the node graph. Write the file; declare the key in `outputs=`; the executor picks it up automatically.

| Output type | File | Helper | Declare in outputs= |
|-------------|------|--------|---------------------|
| Named scalar float | `scalars.json` | `write_scalars(out_dir, key=value, ...)` | `"key": "SCALAR"` |
| 2D field array | `field.npy` | `write_field(out_dir, arr)` | `"field": "FIELD"` |
| Particle positions | `particles.npy` | `write_particles(out_dir, arr)` | `"particles": "PARTICLES"` |
| Mask / alpha | `mask.npy` | `write_mask(out_dir, arr)` | `"mask": "MASK"` |

### Scalars

Write any computed float that downstream nodes might find useful:

```python
from ..core.utils import write_scalars

# After simulation:
write_scalars(out_dir, magnetization=float(M), energy=float(E))
```

Declare: `outputs={"image": "IMAGE", "luminance": "SCALAR", "magnetization": "SCALAR", "energy": "SCALAR"}`

`luminance` is a special scalar — always include it. Compute it as the mean brightness of the output image: `float(np.mean(result))`.

### Fields

A FIELD is a 2D float32 array of arbitrary range — angle fields, potential maps, density grids, gradient magnitudes. It carries spatial structure that downstream nodes (e.g. flow field advection, particle emitters) can consume.

```python
from ..core.utils import write_field

angle_field = compute_angles(...)   # shape (H, W), dtype float32
write_field(out_dir, angle_field)
```

Declare: `"field": "FIELD"`

### Particles

A PARTICLES output is a float32 array of shape (N, 4): `[x, y, vx, vy]` per agent. Downstream nodes (e.g. Particle Painter) consume this to render trails, heatmaps, or point clouds.

```python
from ..core.utils import write_particles

positions = np.stack([x, y, vx, vy], axis=1).astype(np.float32)  # (N, 4)
write_particles(out_dir, positions)
```

Declare: `"particles": "PARTICLES"`

### Masks

A MASK is a 2D float32 array, shape (H, W), values in [0, 1]. 0 = unselected/transparent, 1 = fully selected/opaque. `write_mask` clips to [0, 1] automatically.

**When to expose a mask:** if your method naturally computes "where did I draw, and how much?" — expose that as a mask. Strong candidates:

- Fractals (escape-time): normalized iteration count → smooth gradient mask
- Cellular automata / reaction-diffusion: thresholded `u` concentration or alive/dead grid
- DLA / crystal growth: occupied cells = 1, empty = 0
- Particle simulations: trail density normalized to [0, 1]
- Chladni patterns: nodal lines as binary mask
- Maze: path = 1, wall = 0
- Typography / collage: ink region = 1, background = 0
- Any method with a threshold step: use the pre-threshold float (richer downstream use)

```python
from ..core.utils import write_mask

# Binary: occupied cells
mask = (grid > 0).astype(np.float32)

# Normalized density
mask = np.clip(density / (density.max() or 1), 0, 1)

# Smooth escape-time gradient
mask = (iter_count / max_iter).astype(np.float32)

write_mask(out_dir, mask)
```

Mask quality rules:
- Use float32, not bool. `bool_arr.astype(np.float32)` is fine.
- Match output resolution (H, W). If your internal state is coarser, resize with PIL BILINEAR before writing.
- Write the "positive" mask (selected = 1). The Apply Mask node has an `invert` param.
- Handle the `arr.max() == 0` edge case: `arr / (arr.max() or 1)`.

Declare: `"mask": "MASK"`

---

## Port types

Port types live in an open registry (`core/port_types.py`) — new types are added with `register_port_type()`, no core changes needed.

| Type | Color | Carries |
|------|-------|---------|
| IMAGE | blue | float32 ndarray (H, W, 3), values [0, 1] |
| SCALAR | gray | Python float |
| FIELD | green | float32 ndarray (H, W), arbitrary range |
| PARTICLES | orange | float32 ndarray (N, 4) — [x, y, vx, vy] |
| MASK | white | float32 ndarray (H, W), values [0, 1] |
| COLORMAP | magenta | float32 ndarray (N, 3) or (N, 4) — palette / lookup table |
| ANY | dark gray | wildcard, inputs only |

Type coercion: SCALAR → int param uses `round()`. SCALAR → float passes through. Mismatched types skip silently with a log warning — they do not crash.

---

## What makes a good method

**Be explicit about what you compute.** If your simulation tracks magnetization, energy, or a sync order parameter — expose it as a SCALAR. If it produces a spatial density or angle field — expose it as a FIELD. Another node downstream might use it in ways you haven't anticipated.

**Design for composition.** Your method will be wired to other nodes. Generators (fractals, sims, patterns) produce standalone output and read `input_image` only if composition is meaningful for them. Filters always read and transform `input_image`. Compositing nodes (Blend, Apply Mask) always require typed upstream inputs.

**Keep params expressive.** Params are the user-facing controls. Name them clearly. Add a `description` field. Provide sensible defaults that produce interesting output at zero configuration. Use `choices` lists for categorical params so the UI renders a dropdown automatically.

**Seed everything stochastic.** Use the `seed` argument to initialize `random.seed(seed)` and `np.random.seed(seed)`. Identical seed + params must produce identical output. This makes the system reproducible and lets users lock in results they like.

**Write clearly, not cleverly.** Your code will be read by future agents. Prefer clarity over terseness. A comment explaining why you chose a threshold or constant is worth more than saving a line.

---

## Visibility contract

Every design decision in this repo serves mutual legibility between human and agent:

- **Method files are self-contained.** Reading one file tells you everything about what a node does, what it accepts, and what it produces.
- **Node graph state is JSON.** Every node's method, params, position, render flag, and dirty flag are in the POST body sent to the server. No hidden state.
- **All params are named.** No positional magic. Every param is a key in the `params` dict.
- **Wire payloads are inspectable.** Hovering an edge in the node graph UI shows what's flowing through it.
- **DESIGN.md** is the authoritative architecture document. Update it if you change a core concept.

---

## Pre-flight checklist

Before marking any task done:

1. `uv run python -c "from image_pipeline.server import app"` — must import cleanly, zero errors (a duplicate method id raises here)
2. Every code path produces an image (return-dict, fallback save, or re-raise — see §2)
3. Temp files use `_` prefix
4. `@method` has `outputs=` declared for every sidecar written
5. `@method` has a one-sentence `description=`
6. All helpers are imported explicitly at the top of the file
7. `luminance` scalar is included in `outputs=` and computed as `float(np.mean(result))`
8. `input_image` is guarded with a truthiness check before use
9. `uv run python tools/next_id.py` — get your method ID before writing the file. Never choose one manually.
10. `uv run python tools/audit_methods.py --fail-on-violations` — must exit 0 (this also runs as a pre-commit hook)
11. If you changed a core concept, update `DESIGN.md`
