# Phase 1 Completion Plan — Grillmaster Command Center

This document is a technical spec for the seven infrastructure tasks that complete Phase 1. Phase 1 is not finished when the existing methods work — it's finished when the **infrastructure for adding the next 1000 methods is solid**. Each task below is scoped to be independently implementable.

---

## Task 1 — Graph Persistence (Save / Load)

**Why:** Without save/load, every session starts from scratch. This is the minimum viable workflow for development and the most impactful missing piece.

### Data format

```json
{
  "version": 1,
  "nodes": [
    {
      "id": "node-abc123",
      "methodId": 42,
      "params": { "scale": 1.5, "palette": "fire" },
      "x": 320,
      "y": 180,
      "render": false
    }
  ],
  "edges": [
    {
      "from": "node-abc123",
      "fromPort": "image",
      "to": "node-def456",
      "toPort": "image_in"
    }
  ]
}
```

### Frontend (ui/index.html)

- Add a **Save** button to the graph top bar. On click: serialize current `gNodes` + `gEdges` to the JSON format above, offer as a `.grillmaster` file download.
- Add a **Load** button. On click: open a file picker, parse the JSON, call `gLoadGraph(data)` which clears the canvas and rebuilds nodes/edges from the saved state.
- **Auto-save to localStorage** on every graph change (debounced 2000ms). Key: `gm-autosave`. On page load, if `gm-autosave` exists, offer a "Restore last session?" banner.
- **Named sessions panel** (optional, stretch): a sidebar listing graphs saved server-side, clickable to load.

### Backend (server.py)

```
POST /api/graph/save          body: { name: string, graph: GraphJSON }
GET  /api/graph/saved         → list of { name, saved_at } 
GET  /api/graph/saved/{name}  → GraphJSON
DELETE /api/graph/saved/{name}
```

Saved graphs live at `OUTPUT_ROOT / "saved-graphs" / "{name}.json"`.

### Files to change
- `ui/index.html` — Save/Load buttons, `gLoadGraph()`, auto-save logic
- `image_pipeline/server.py` — four new endpoints

---

## Task 2 — Method Hot-Reload

**Why:** Adding a new method currently requires killing and restarting the server. For a pipeline that grows perpetually this friction compounds badly.

### Approach

Use the `watchdog` Python library to watch `image_pipeline/methods/` recursively. On any `.py` file creation or modification:

1. Determine which module corresponds to the changed file.
2. Clear the old registration from the registry (`registry.unregister(id)`).
3. `importlib.reload(module)` the changed file, which re-executes the `@method` decorator and re-registers.
4. Emit a Server-Sent Event to all connected graph clients: `event: node-defs-updated\ndata: {}\n\n`
5. Frontend receives the event, re-fetches `/api/node-defs`, updates the Tab menu and any open param panels.

### Registry changes (core/registry.py)

Add:
```python
def unregister(method_id: int):
    """Remove a method from the registry (used by hot-reload)."""
    _REGISTRY.pop(method_id, None)
```

### Server changes (image_pipeline/server.py)

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MethodFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            _hot_reload(event.src_path)
    on_created = on_modified

# Start observer in lifespan or startup event
```

Also add a global `asyncio.Queue` for SSE broadcast and a `/api/events` endpoint that streams from it. The `node-defs-updated` event goes into this queue on hot-reload.

### Frontend (ui/index.html)

```javascript
const evtSource = new EventSource('/api/events');
evtSource.addEventListener('node-defs-updated', () => {
    gLoadMethodPalette();  // re-fetch /api/node-defs
});
```

### Dependencies
- `watchdog` must be added to `requirements.txt`

### Files to change
- `image_pipeline/core/registry.py` — add `unregister()`
- `image_pipeline/server.py` — watchdog observer, SSE broadcast queue, `/api/events`
- `ui/index.html` — EventSource listener for `node-defs-updated`

---

## Task 3 — Port Type Registry (Extensibility)

**Why:** Port types are currently a hardcoded enum in `graph.py`. Adding GEOMETRY (Phase 6) or any future type requires editing core. New types should be declarable without touching the executor.

### New file: core/port_types.py

```python
from dataclasses import dataclass, field

@dataclass
class PortTypeSpec:
    name: str
    color: str                          # hex color used in the UI
    description: str                    # shown in tooltips
    accepts_from: list[str] = field(default_factory=list)  # coercible input types

_PORT_REGISTRY: dict[str, PortTypeSpec] = {}

def register_port_type(
    name: str,
    color: str,
    description: str,
    accepts_from: list[str] | None = None,
) -> None:
    _PORT_REGISTRY[name] = PortTypeSpec(
        name=name,
        color=color,
        description=description,
        accepts_from=accepts_from or [],
    )

def get_port_type(name: str) -> PortTypeSpec | None:
    return _PORT_REGISTRY.get(name)

def all_port_types() -> dict[str, PortTypeSpec]:
    return dict(_PORT_REGISTRY)

# Built-in registrations
register_port_type("IMAGE",     "#4a9eff", "float32 ndarray (H,W,3) values [0,1]")
register_port_type("SCALAR",    "#888888", "Python float", accepts_from=["IMAGE"])
register_port_type("FIELD",     "#4caf50", "float32 ndarray (H,W) arbitrary range")
register_port_type("PARTICLES", "#ff9800", "float32 ndarray (N,4) — [x,y,vx,vy]")
register_port_type("MASK",      "#e8e8e8", "float32 ndarray (H,W) values [0,1]")
register_port_type("ANY",       "#444444", "wildcard input type")
```

### graph.py changes

Remove the `PortType` enum. Replace references with string comparisons against the registry:

```python
from .port_types import get_port_type, all_port_types

# Instead of: port_type == PortType.IMAGE
# Use:        port_type == "IMAGE"
```

### server.py — new endpoint

```
GET /api/port-types  →  { "IMAGE": { "color": "#4a9eff", ... }, ... }
```

### Frontend (ui/index.html)

Replace the hardcoded port color map with a fetch from `/api/port-types` on init. New port types automatically get their correct colors.

### Files to change
- `image_pipeline/core/port_types.py` — new file
- `image_pipeline/core/graph.py` — remove `PortType` enum, use string keys
- `image_pipeline/server.py` — `/api/port-types` endpoint
- `ui/index.html` — dynamic port color map from API

---

## Task 4 — Automated Method Audit (CI Gate)

**Why:** `tools/audit_methods.py` runs manually. For a perpetually growing library, violations must be caught on commit, not discovered weeks later.

### Extend audit_methods.py

Add a `--fail-on-violations` flag: exit code 1 if any method has a violation. This makes it usable as a CI/pre-commit gate.

```bash
uv run python tools/audit_methods.py --fail-on-violations
# exits 0 if clean, 1 if violations found
```

Existing checks: missing `outputs=`, missing `luminance`.

Add new checks:
- **Sidecar/declaration mismatch**: method calls `write_field()` but `outputs=` has no `"FIELD"` key (and vice versa). Use AST to detect calls to `write_field`, `write_particles`, `write_mask`, `write_scalars`.
- **No PNG fallback**: method has an `except` block with no `save_image` call inside it.
- **ID collision**: two methods share the same `id=` value — hard error.
- **Missing description**: `description=""` or absent — warning only (not a hard violation).

### Pre-commit hook

Add `.pre-commit-config.yaml` to the repo root:

```yaml
repos:
  - repo: local
    hooks:
      - id: audit-methods
        name: Grillmaster method audit
        entry: uv run python tools/audit_methods.py --fail-on-violations
        language: system
        pass_filenames: false
        files: ^image_pipeline/methods/
```

Install with: `pre-commit install`

Alternatively, add a GitHub Actions workflow at `.github/workflows/audit.yml` if the repo uses CI.

### Files to change
- `tools/audit_methods.py` — `--fail-on-violations` flag, new checks (sidecar mismatch, PNG fallback, ID collision)
- `.pre-commit-config.yaml` — new file

---

## Task 5 — Error Visibility in the Graph

**Why:** A method that crashes produces a silent blank output. In a multi-node graph with one broken node in the middle, this is nearly undebuggable.

### Backend (graph.py)

In the node execution loop, wrap each node run in a try/except. On failure:

1. Write a fallback PNG (red X or dark placeholder) to `node_dir` so downstream nodes don't also fail.
2. Emit an SSE event: `event: node-error\ndata: {"nodeId": "...", "error": "short traceback"}\n\n`
3. Collect errors in a dict and include in the final run response: `{ "errors": { "node-abc": "ZeroDivisionError: ..." } }`

```python
try:
    method_fn(out_dir=node_dir, seed=node.seed, params=run_params)
except Exception as e:
    import traceback
    err_summary = traceback.format_exc(limit=5)
    # write fallback PNG
    _write_error_placeholder(node_dir)
    # emit SSE
    yield f"event: node-error\ndata: {json.dumps({'nodeId': node_id, 'error': err_summary})}\n\n"
    node_errors[node_id] = err_summary
```

### Frontend (ui/index.html)

- SSE handler for `node-error`: add class `node-error` to the node element with that ID.
- `.node-error` CSS: red/dark border, a `⚠` badge in the top-right of the node header.
- Tooltip on the badge shows the truncated error message (first 3 lines of traceback).
- On run start, clear all `node-error` classes.
- Node body shows "Error — see badge" instead of a stale output preview when in error state.

### Files to change
- `image_pipeline/core/graph.py` — try/except per node, SSE error events, `_write_error_placeholder()`
- `ui/index.html` — `node-error` CSS class, badge rendering, SSE handler

---

## Task 6 — Method Metadata Richness

**Why:** The Tab menu search quality and agent legibility both depend on rich method metadata. Description, version, and deprecated are the three fields with immediate payoff.

### registry.py — extend MethodMeta

```python
@dataclass
class MethodMeta:
    name: str
    id: int
    tags: list[str]
    category: str
    outputs: dict[str, str] | None = None
    inputs: dict[str, str] | None = None
    description: str = ""       # human-readable summary, shown in Tab menu + node tooltip
    version: int = 1            # increment when params change in a breaking way
    deprecated: bool = False    # hide from Tab menu, show warning badge in graph
```

### @method decorator

Accept and pass through the new kwargs:

```python
@method(
    name="Reaction Diffusion",
    id=32,
    tags=["simulation"],
    description="Gray-Scott reaction-diffusion system. Produces organic spotted and striped patterns.",
    version=2,
    outputs={ ... },
)
```

All new kwargs have safe defaults — existing method files need no immediate changes.

### /api/node-defs response

Include all new fields in the JSON response so the frontend can use them.

### Frontend (ui/index.html)

- **Tab menu**: show `description` as a subtitle line below the method name (truncated to ~60 chars, ellipsis).
- **Node tooltip**: full description shown on hover over the node header.
- **Deprecated nodes**: render with a yellow border and strikethrough on the method name. Exclude from Tab menu results by default; show a "(deprecated)" section at the bottom if the search string explicitly matches.
- **Version badge**: optional — show in node header as a small `v2` label (useful when a method has been audited/updated).

### Files to change
- `image_pipeline/core/registry.py` — extend `MethodMeta`, extend `@method` decorator
- `image_pipeline/server.py` — include new fields in `/api/node-defs`
- `ui/index.html` — Tab menu subtitle, node tooltip, deprecated styling

---

## Task 7 — Unique ID Enforcement

**Why:** Agents are currently told to "grep the repo" for an unused ID. Two agents working in parallel will collide. ID collisions cause silent registry overwrites and are very hard to debug.

### New tool: tools/next_id.py

```python
#!/usr/bin/env python3
"""
Print the next available method ID.
Usage: uv run python tools/next_id.py
       uv run python tools/next_id.py --reserve 5   # claim next 5 IDs
"""
import ast, sys
from pathlib import Path

def get_used_ids() -> set[int]:
    ids = set()
    for f in Path("image_pipeline/methods").rglob("*.py"):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                        ids.add(int(kw.value.value))
    return ids

def main():
    reserve = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == '--reserve' else 1
    used = get_used_ids()
    next_id = max(used) + 1 if used else 1
    if reserve == 1:
        print(f"Next available method ID: {next_id}")
    else:
        ids = list(range(next_id, next_id + reserve))
        print(f"Next {reserve} available IDs: {ids}")

if __name__ == "__main__":
    main()
```

### Collision detection in audit

`tools/audit_methods.py` already walks all methods. Add:

```python
# Check for ID collisions
id_to_files = defaultdict(list)
for method in all_methods:
    id_to_files[method.id].append(method.file)

for method_id, files in id_to_files.items():
    if len(files) > 1:
        violations.append(f"ID COLLISION: id={method_id} used in {files}")
```

ID collisions are hard errors — `--fail-on-violations` always exits 1 when any are present.

### AGENT_GUIDE.md update

Add to the pre-flight checklist:
> `uv run python tools/next_id.py` — get your method ID before writing the file. Never choose one manually.

### Files to change
- `tools/next_id.py` — new file
- `tools/audit_methods.py` — collision detection
- `AGENT_GUIDE.md` — reference `next_id.py` in checklist

---

## Implementation order

Tasks are largely independent but the suggested order for minimal rework:

1. **Task 7** (ID enforcement) — no dependencies, high value immediately, prevents future pain
2. **Task 6** (metadata richness) — no dependencies, unlocks better Tab menu and deprecation
3. **Task 3** (port type registry) — refactor before more port types accumulate
4. **Task 5** (error visibility) — unblocks debugging of all other tasks
5. **Task 4** (audit CI gate) — more useful once metadata and IDs are clean
6. **Task 1** (graph persistence) — highest user-facing impact, save for when core is stable
7. **Task 2** (hot-reload) — most complex, do last; depends on registry being stable
