# Skill Update — Grillmaster Command Center

**Read this before writing or modifying any method file in this repo.**

This prompt brings your understanding of the grillmaster-command-center pipeline up to date. It supersedes any prior knowledge you have about this system. Read it fully, then apply every rule it describes to your work.

---

## What this system is

Grillmaster Command Center is a node-based generative image pipeline designed to be extended by LLM agents. It is a Python-native analog to Houdini: nodes produce typed outputs, those outputs flow through wires, downstream nodes read what they need by name. The current pipeline is 2D (image generation). The architecture is explicitly designed to generalize to 3D — do not hardcode 2D-specific assumptions deeper than the image I/O layer.

Every design decision is made with **mutual legibility** in mind: a human reading a method file should understand it completely, and so should an agent. Self-contained, explicit, no magic.

---

## What has changed — read carefully

### 1. No hidden compositing

The executor no longer applies a post-hoc overlay blend to method outputs. Previously, ~132 of 160 methods had their output silently composited with the upstream image. **That behavior is gone.** 

What this means for you:
- Whatever your method writes to its output PNG is exactly what flows downstream. No overlay. No blending.
- If your method supports reading an upstream image, it reads from `params.get("input_image", "")` and handles that itself.
- If your method does not use upstream images, write your output and stop. Do not blend anything.

### 2. `@method` now requires `outputs=` and supports `inputs=`

Every method must declare its outputs:

```python
@method(
    name="My Method",
    id=42,
    tags=["simulation"],
    outputs={
        "image": "IMAGE",
        "luminance": "SCALAR",
        # declare every sidecar you write:
        # "field": "FIELD",
        # "magnetization": "SCALAR",
        # "mask": "MASK",
    },
    # declare typed wire inputs if applicable:
    # inputs={"particles": "PARTICLES"},
)
def run(out_dir, seed, params):
    ...
```

If you are updating an existing method that lacks `outputs=`, add it. At minimum: `{"image": "IMAGE", "luminance": "SCALAR"}`.

### 3. New port types: MASK and the full sidecar protocol

Six port types now exist:

| Type | Carries |
|------|---------|
| IMAGE | float32 ndarray (H,W,3), values [0,1] |
| SCALAR | Python float |
| FIELD | float32 ndarray (H,W), arbitrary range |
| PARTICLES | float32 ndarray (N,4) — [x, y, vx, vy] |
| MASK | float32 ndarray (H,W), values [0,1] |
| ANY | wildcard, inputs only |

Sidecar helpers — import from `..core.utils`:

```python
write_scalars(out_dir, key=float_value, ...)  # → scalars.json
write_field(out_dir, arr)                      # → field.npy   (H,W) float32
write_particles(out_dir, arr)                  # → particles.npy (N,4) float32
write_mask(out_dir, arr)                       # → mask.npy   (H,W) float32, auto-clipped [0,1]
```

Write the sidecar. Declare it in `outputs=`. The executor picks it up automatically.

### 4. Add a mask output where it makes sense

If your method computes a meaningful spatial selection — where it drew, what's alive, what's occupied — expose it as a mask. Strong candidates:

- Fractals: `(iter_count / max_iter).astype(np.float32)`
- Cellular automata / reaction-diffusion: `np.clip(u, 0, 1)` or `(grid > 0).astype(np.float32)`
- DLA / crystal / dendrite: occupied cells = 1
- Particle simulations: trail density normalized to [0,1]
- Maze: path = 1, wall = 0
- Any method with a threshold step: use the pre-threshold float

Rules: float32 not bool, same (H,W) as output image, write positive mask (selected=1), handle `arr.max() == 0`.

### 5. Metadata fields are expanding

The `@method` decorator will soon support (use now where you can):

```python
@method(
    name="...",
    id=42,
    tags=[...],
    description="One sentence describing what this method produces.",  # shown in UI
    version=1,       # increment if you change params in a breaking way
    deprecated=False,
    outputs={...},
)
```

`description` is the most important new field. Add it to every method you touch.

### 6. Method IDs are permanent and must be unique

Never reuse an ID. Never choose one manually. Run:

```bash
uv run python tools/next_id.py
```

This scans the repo and returns the next safe ID. If `next_id.py` does not exist yet, grep for the highest `id=` value across all method files and use `max + 1`.

---

## Full method contract

### Always write a PNG on every code path

```python
try:
    result = compute(...)
    save_image(result, out_dir / f"{METHOD_ID:04d}_output.png")
except Exception as e:
    print(f"[warn] {e}, falling back")
    blank = np.zeros((height, width, 3), dtype=np.float32)
    save_image(blank, out_dir / f"{METHOD_ID:04d}_fallback.png")
```

Every branch — including early returns and except blocks — must write a PNG.

### Temp files use `_` prefix

```python
tmp = out_dir / "_intermediate.png"   # excluded from output detection
```

### Imports must be explicit

```python
from ..core.utils import save_image, apply_palette, write_scalars, write_field, write_particles, write_mask
from ..core.animation import capture_frame
```

Never call a helper without importing it first.

### Handle `input_image` gracefully

```python
input_path = params.get("input_image", "")
if input_path and Path(input_path).exists():
    img = np.array(Image.open(input_path)).astype(np.float32) / 255.0
else:
    img = np.zeros((height, width, 3), dtype=np.float32)
```

### Seed everything stochastic

```python
random.seed(seed)
np.random.seed(seed)
```

Same seed + same params = same output, always.

### Luminance is always required

```python
write_scalars(out_dir, luminance=float(np.mean(result)))
```

Include `"luminance": "SCALAR"` in `outputs=`.

---

## Self-check procedure

Run these checks on every method you write or modify before calling the task done.

### Step 1 — Import check
```bash
cd /Users/admin/Documents/GitHub/grillmaster-command-center
uv run python -c "from image_pipeline.server import app"
```
Must complete with zero errors. If it fails, fix imports before proceeding.

### Step 2 — Smoke test the method
```bash
uv run python -c "
from image_pipeline.core.registry import get_registry
from pathlib import Path
import tempfile, random, numpy as np

reg = get_registry()
m = reg[YOUR_METHOD_ID]   # replace with actual id
with tempfile.TemporaryDirectory() as d:
    m.fn(out_dir=Path(d), seed=42, params={})
    pngs = [f for f in Path(d).iterdir() if f.suffix == '.png' and not f.name.startswith('_')]
    assert pngs, 'NO PNG WRITTEN'
    print('OK:', pngs[0].name)
"
```
Must produce at least one non-`_`-prefixed PNG.

### Step 3 — Declaration audit
```bash
uv run python tools/audit_methods.py
```
Your method must appear clean in the report. Violations to fix: missing `outputs=`, missing `luminance`, undeclared sidecars, ID collision.

### Step 4 — Manual checklist

- [ ] `outputs=` declared on `@method` with every sidecar listed
- [ ] `luminance` in `outputs=` and written via `write_scalars`
- [ ] Every code path (including except blocks) writes a PNG
- [ ] Temp files use `_` prefix
- [ ] All helpers explicitly imported at top of file
- [ ] `input_image` guarded with truthiness check before use
- [ ] Method ID is unique (verified with `tools/next_id.py` or grep)
- [ ] `description=` added to `@method`
- [ ] `random.seed(seed)` and `np.random.seed(seed)` called at the top of `run()`

---

## Simulations & animated methods — extra contract

If the method simulates or animates (its output evolves over the timeline), the
above is necessary but **not sufficient**. It must also play correctly and stay
fast in all three render contexts (single still, clip, live) — most importantly,
**per-frame cost must never grow as the timeline advances**. A stateless node
that re-simulates up to `time` every frame gets slower and slower forever (this
was the Cellular Automata #18 bug). A state-based sim must instead keep its
**last state** and step it one step per frame — the **persistent stateful
pattern** (runs forever at constant cost; #18 works this way now) — or, for a
finite scrubbable sequence, the cook-a-window Architecture-A pattern.

Before writing or rewriting any simulation node, read and apply
**`docs/prompts/simulation-node-render-contract.md`** — the Architecture A/B
decision, the render-context matrix, the `-1.0` scalar-override sentinel rule,
the "show-from-start" rule, and the no-slowdown verification procedure.

---

## Where to find the full spec

- `docs/prompts/simulation-node-render-contract.md` — **simulation / animation node contract** (arch A/B, performance, render-system integration)
- `AGENT_GUIDE.md` — complete method authoring guide with examples
- `DESIGN.md` — authoritative architecture document (see "Live mode" for the render invariants)
- `image_pipeline/core/arch.py` — Architecture A/B detection
- `image_pipeline/core/utils.py` — all helper functions
- `image_pipeline/core/registry.py` — `@method` decorator definition
- `image_pipeline/methods/compositing/` — reference implementations for typed I/O nodes
