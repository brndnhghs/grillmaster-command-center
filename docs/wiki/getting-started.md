# Getting Started

## Prerequisites

- **Python 3.12** — pinned via `.python-version`; the project is built and tested on 3.12 on both macOS and Windows.
- **uv** — the venv is uv-managed (`uv venv .venv`). The repo's `.python-version` makes uv pick 3.12 automatically.
- **Git** — to clone, and to track the source SHA.
- **Node.js 22+** — only for the 3D viewport sidecar (`image_pipeline/3d/threejs-sidecar.mjs`), which the server spawns automatically when a graph uses 3D nodes. The 2D pipeline needs no Node.
- **NPM** — `gl` (headless WebGL) must be installed once: `npm install gl` in the repo root (the sidecar renders via ANGLE/D3D11 on Windows, Metal/ANGLE on macOS).
- **GPU** — Apple (Metal/OpenGL) and NVIDIA (OpenGL/CUDA) are supported. GPU shader methods use `moderngl` (OpenGL 3.3+). On hybrid Intel+NVIDIA laptops, WGL defaults to the Intel adapter; update the NVIDIA driver (≥430) so Windows' per-app GPU preference can route GL to the NVIDIA GPU.

## Installation

```bash
git clone https://github.com/brndnhghs/grillmaster-command-center.git
cd grillmaster-command-center

# Create the venv (uv reads .python-version → 3.12)
uv venv .venv

# Install the pinned dependency set
uv pip install -r requirements.txt            # macOS / Linux
uv pip install -r requirements.txt --python .venv/Scripts/python.exe   # Windows
```

`requirements.txt` pins the known-good set (regenerated 2026-06-20):

| Package | Version | Role |
|----------|---------|------|
| `fastapi` | 0.137.2 | Web framework (both servers) |
| `uvicorn` | 0.47.0 | ASGI server |
| `numpy` | 2.4.6 | Array math for image tensors |
| `opencv-python` | 4.13.0.92 | Post-process filters, compositing |
| `Pillow` | 12.2.0 | Image read/write |
| `pydantic` | 2.13.4 | Request/response validation |
| `PyYAML` | 6.0.3 | Preset loading |
| `pyngrok` | 8.1.2 | Localhost tunneling (optional) |
| `watchdog` | >=4.0.0 | File-system watcher for method hot-reload |
| `moderngl` | — | GPU shader methods (#82, gpu_shaders category) |
| `pyfiglet` | 1.0.4 | ASCII-art methods (hard-imported — required) |

**Optional extras** (commented out in `requirements.txt`; each is imported lazily inside the method that needs it, so the server runs without them — only that method fails until installed): `matplotlib` (colormaps), `scikit-image` (fractal resize), `qrcode` (#09 QR Code), `torch` + `diffusers` (#21 Stable Diffusion 1.5, ~2 GB+).

## First Run

The simplest path is the Dashboard, which launches and monitors both services:

```bash
# macOS / Linux
source .venv/bin/activate
python -m dashboard --autostart

# Windows (git-bash or cmd)
.venv/Scripts/python.exe -m dashboard --autostart
# → http://127.0.0.1:7870
```

Open `http://127.0.0.1:7870` in a browser. The dashboard shows Launch/Stop controls for each service and an embedded UI switcher.

**Alternative — launcher scripts** (repo-relative, portable):

```bash
bash scripts/dashboard.sh              # macOS / Linux / git-bash
bash scripts/grillmaster-launcher.sh   # pipeline only
scripts/launch_pipeline.bat            # Windows: clickable pipeline launcher
```

> **Agent-shell gotcha (Windows):** shells launched from agent runtimes (e.g. Hermes) export a `PYTHONPATH` pointing at the agent's own site-packages, which shadows the repo venv and breaks numpy/fastapi versions. The `.bat` launchers and `grillmaster-launcher.sh` clear `PYTHONPATH` before starting; if you launch by hand from such a shell, run `env -u PYTHONPATH .venv/Scripts/python.exe -m image_pipeline.server`.

## Service Ports

| Service | Port | Launched by |
|----------|-------|-------------|
| Image Pipeline (server) | `7860` | Dashboard / launcher |
| 3D Sidecar (Node.js) | `7862` | Image Pipeline (on demand) or Dashboard |
| Dashboard (control panel) | `7870` | you |

You can also run a single service directly:

```bash
.venv/bin/python -m image_pipeline.server --port 7860          # macOS / Linux
.venv/Scripts/python.exe -m image_pipeline.server --port 7860   # Windows
```

## Common Workflows

### Generate a single method (CLI)
```bash
python -m image_pipeline.pipeline --all
python -m image_pipeline.pipeline --group fractals --parallel 4
python -m image_pipeline.pipeline --methods 07,21,49 --composite overlay
```

### Generate a single method (UI)
Open the Dashboard → Image Pipeline → **Methods** tab. Search a method, tweak params, click **Generate**. Watch progress in the output panel; download when done.

### Build & run a node graph
Open the **Node Graph** tab. Drag methods from the palette onto the canvas, wire outputs→inputs, set canvas size, then **Run**. Use **Live** for a continuous simulation that absorbs edits without restarting.

## Configuration

- **`GRILLMASTER_API_TOKEN`** — when set, the server requires this token on protected endpoints (Node Doctor apply/undo, Node Tester batch-apply). The UI reads it from `localStorage['api-token']` and attaches it as `X-Api-Token` on every request. No-op when unset (local/dev).
- **`THREEJS_SIDECAR_URL`** — default `http://127.0.0.1:7862`. The server proxies 3D-node graph renders to this Node.js sidecar, spawning it on first use if nothing is listening. Override to point at a remote sidecar.
- **`THREEJS_SIDECAR_EXTERNAL`** — set to `1` when something else owns the sidecar process (a supervisor, a debugger, a remote host). The server will then proxy to it but never spawn it.
- **`HERMES_AGENT_DIR`** / **`HERMES_PYTHON`** — Node Doctor backend (default `~/.hermes/hermes-agent`).
- **`data/logs/`** — service stdout/stderr are written here (e.g. `data/logs/pipeline.log`). Useful when a launch reports "failed".

## Regenerating the glyph atlases

The ASCII shader (`ascii_art_gpu`) requires `image_pipeline/core/glyph_atlas_*.png` (6 fonts). They are committed; regenerate with:

```bash
python image_pipeline/tools/build_glyph_atlas.py   # finds macOS or Windows fonts
```

## Where to Go Next

- Architecture: [architecture.md](architecture.md)
- Module reference: [README.md#module-map](README.md#module-map)
- HTTP API index: [api.md](api.md)
