# Configuration — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Environment Variables

| Env Var | Purpose | Default |
|---------|---------|---------|
| `HERMES_AGENT_DIR` | Hermes agent install for Node Doctor | Auto-detected (see below) |
| `HERMES_PYTHON` | Override interpreter for Hermes runner | Auto-detected (see below) |
| `GRILLMASTER_API_TOKEN` | API token for mutating endpoints | Empty (no auth) |

### Hermes auto-detection

Hermes is the sole LLM backend (Node Doctor, Node Tester fixes), so the server
resolves it at startup and logs which interpreter it found. Setting either
variable always overrides detection.

Search order for the checkout, first match wins:

1. `$HERMES_AGENT_DIR`
2. `~/.hermes/hermes-agent`
3. `%LOCALAPPDATA%/hermes/hermes-agent` (Windows installer default)
4. `~/AppData/Local/hermes/hermes-agent`
5. `~/hermes/hermes-agent`

A candidate only matches if it actually contains a venv interpreter. Both venv
layouts are probed — `venv/Scripts/python.exe` (Windows) and `venv/bin/python`
(POSIX) — so the same configuration works on either platform.

If nothing is found the startup log says so and names `~/.hermes/hermes-agent`,
which is the path to point `HERMES_AGENT_DIR` at.

## Requirements

### Python (requirements.txt)
- `fastapi==0.137.2` — Web framework
- `uvicorn==0.47.0` — ASGI server
- `numpy==2.4.6` — Numerical computing
- `scipy==1.17.1` — Scientific computing
- `opencv-python==4.13.0.92` — Image processing + JPEG encode
- `Pillow==12.2.0` — Image I/O
- `pydantic==2.13.4` — Data validation
- `PyYAML==6.0.3` — YAML config
- `pyngrok==8.1.2` — Tunneling
- `watchdog>=4.0.0` — File watcher (hot-reload)
- `moderngl` — GPU shaders (commented out, optional)

### Node.js (package.json)
- `three@^0.185.1` — 3D viewport
- `puppeteer@^25.3.0` — Browser automation
- `gl@^8.1.6` — WebGL

## Files

| File | Purpose |
|------|---------|
| `pytest.ini` | Test configuration (slow marker excluded by default) |
| `.pre-commit-config.yaml` | Pre-commit hook (method audit) |
| `image_pipeline/config/groups.yaml` | Method grouping for UI |

## Runtime Directories

| Path | Purpose |
|------|---------|
| `image_pipeline/output/` | Generated images, sequences, saved graphs, logs |
| `image_pipeline/output/graphs/` | Graph document persistence |
| `image_pipeline/output/saved-graphs/` | Named graph saves |
| `image_pipeline/output/sequences/` | Rendered sequences |
| `image_pipeline/output/assets/` | User-uploaded 3D models |
| `~/.cache/image-pipeline/` | Legacy method cache (CLI-only) |