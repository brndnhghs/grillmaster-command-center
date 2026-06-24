# Pipeline Design

## Philosophy

The grillmaster pipeline is a **named-attribute payload model** — inspired by Houdini's node geometry model. Each node produces a structured dict of typed outputs (its _payload_), not just a single image. Downstream nodes can consume any attribute from the payload, with attributes propagating automatically unless overridden.

Three principles drive all design decisions:

1. **Non-destructive composition.** Nodes read upstream outputs; they never mutate them. Every node has its own output directory, so the full graph state is always on disk and auditable.

2. **Context separation.** Generation parameters live in `params`, wiring lives in `edges`, execution policy lives in the executor. Methods know nothing about the graph; the graph knows nothing about image formats.

3. **Named attributes as the contract.** A wire carries a named attribute — `luminance`, `r`, `field`, `particles` — not an anonymous blob. Name-based scoring means wires can be re-routed automatically when ports are renamed or method signatures evolve.

---

## Port Type System

| Type       | Value key    | Wire colour | Carries                                   |
|------------|--------------|-------------|-------------------------------------------|
| `IMAGE`    | `image`      | white        | H×W×3 float32 ndarray, range [0, 1]      |
| `SCALAR`   | `luminance` / any named scalar | yellow | float                  |
| `FIELD`    | `field`      | blue         | H×W float32 ndarray (angle / potential)  |
| `PARTICLES`| `particles`  | orange       | N×4 float32 [x, y, vx, vy]              |
| `ANY`      | —            | grey         | fallback; accepts any upstream value      |

Port types are declared in `PortType(str, Enum)` in `core/graph.py`. String equality lets them serialize cleanly to JSON and compare across the graph/registry boundary.

---

## Sidecar Protocol

Methods write sidecar files alongside their PNG to expose non-image outputs:

| File             | Content                          | Produced by                     |
|------------------|----------------------------------|---------------------------------|
| `scalars.json`   | `{"key": float, …}`              | `write_scalars(out_dir, {…})`   |
| `field.npy`      | H×W float32 ndarray              | `write_field(out_dir, arr)`     |
| `particles.npy`  | N×4 float32 [x, y, vx, vy]      | `write_particles(out_dir, arr)` |

The executor reads these back after each node execution and merges them into `flat_outputs[node_id]`. Because they live on disk, the executor can reload them without re-running the method — enabling the dirty-flag skip.

Helper functions live in `core/utils.py`. Methods import the helpers they need.

---

## Execution Model

`GraphExecutor.execute()` performs a single frame pass:

1. **Topological sort** (Kahn's algorithm, feedback edges excluded).
2. **Truncate at terminal** — the node flagged `render=True` or the last node with no outgoing edges.
3. **For each node in order:**
   a. **Dirty check**: if `node.dirty == False` and no upstream node ran this frame, load cached PNG + sidecars from disk → skip re-execution.
   b. **Upstream scalar harvest**: collect all scalar attributes from connected upstream payloads; name-score them against the current node's params and pre-populate `run_params` (implicit inheritance).
   c. **Edge processing**: explicit wires override the pre-seeded values. IMAGE → `input_image` file path; PARTICLES → direct `run_params[dst_port]`; SCALAR/FIELD → typed injection with `_inject_typed`.
   d. **Execute** `meta.fn(node_dir, seed, params=run_params)`.
   e. **Read back**: PNG + sidecars → `flat_outputs[node_id]`.
   f. **Payload inheritance**: merge upstream scalars (lower priority) into `flat_outputs[node_id]` so downstream nodes can read inherited attributes even without explicit scalar wires.
4. Store `flat_outputs` as `_prev_outputs` for feedback edges.

The executor instance uses `_GRAPH_SESSION_DIR` (a stable directory) so that cached outputs persist across graph runs. Per-run directories (`graph-{job_id}`) are only used for the final assembled output file.

---

## Output Declarations

Every method declares its outputs in the `@method` decorator:

```python
@method(
    id="34",
    outputs={"image": "IMAGE", "luminance": "SCALAR", "particles": "PARTICLES"},
)
```

The executor reads `meta.outputs` via `_make_node_def()` to build the port list. No tag-based guessing. Named sidecar scalars (e.g. `r`, `amplitude`, `spread`) are registered via `write_scalars` at runtime and appear in `flat_outputs` automatically — they don't need explicit `outputs=` entries (though declaring them is good practice for discoverability).

Input ports are auto-derived from param defaults:
- `int` / `float` default → SCALAR input port
- `list` / `tuple` default → FIELD input port

Ports that can't be auto-derived (e.g. a PARTICLES input whose param defaults to `None`) are declared with `inputs={"particles": "PARTICLES"}` in the decorator.

---

## Current Named Outputs Table

| Method id | Name                   | `outputs` keys                                  |
|-----------|------------------------|-------------------------------------------------|
| 16        | Flow Field (codegen)   | image, luminance, field                         |
| 20        | Particles              | image, luminance, particles                     |
| 34        | Boids                  | image, luminance, particles                     |
| 35        | Flowfield              | image, luminance, field                         |
| 83        | Langton's Ant          | image, luminance, particles                     |
| 86        | Physarum               | image, luminance, field, particles              |
| 88        | Particle Life          | image, luminance, particles                     |
| 106       | Dielectric Breakdown   | image, luminance, field                         |
| 113       | N-body Gravity         | image, luminance, field                         |
| 130       | Particle Painter       | image, luminance _(PARTICLES consumer)_         |

All other methods produce at minimum `image` and `luminance` (the default).

---

## Planned Extensions

### Wire Inspector
Hovering a bezier edge shows a tooltip listing the upstream node's payload manifest (keys + types). Implemented: `GET /api/graph/wire-payload/{job_id}/{src_node_id}` reads from `_GRAPH_SESSION_DIR / node_id`, the frontend fetches on `mouseenter` and positions a fixed `#wire-tooltip` div.

### Dirty Flag / Selective Recooking
`GraphNode.dirty: bool = True`. The executor skips a node if `dirty=False` and no upstream node ran. The frontend marks a node dirty when any param changes; resets all nodes to clean after a successful run. This makes iterative tweaking fast — only changed nodes re-execute.

### Named Image Planes
Future: methods could write `beauty.npy`, `depth.npy`, `normals.npy` alongside the main PNG (analogous to Houdini deep / mantra planes). The executor would expose them as additional IMAGE-type outputs.

### VEX-style Wrangle Node
A node that accepts a small Python/expression snippet and runs it over the upstream payload dict — analogous to Houdini's Attribute Wrangle. Useful for ad-hoc scalar transforms without a full method file.

---

## Visibility Contract

- **Methods** write to `out_dir` and return an `Image.Image`. They must not read from sibling node directories. They must not import from `core/graph.py`.
- **The executor** constructs `run_params` from node params + upstream wires. It must not know about image formats beyond RGB PNG.
- **The server** serialises/deserialises the graph JSON and manages job lifecycle. It must not hold long-lived numpy arrays in memory.
- **The frontend** is the only place where port colours, node layout, and edge routing are decided. The backend never sends pixel coordinates.
