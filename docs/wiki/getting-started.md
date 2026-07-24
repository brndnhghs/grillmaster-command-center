# Getting Started

## Prerequisites

- **Python 3.12** — the repo ships a `.venv` built on 3.12.13. The Image Pipeline and Chord Bot both run under it.
- **Git** — to clone, and to track the source SHA.
- **Node.js** — only for the 3D viewport sidecar (`image_pipeline/3d/threejs-sidecar.mjs`). The 2D pipeline needs no Node.
- **macOS / Linux** — paths and `lsof`/`killpg` usage assume a Unix-like shell.

## Installation

```bash
git clone https://github.com/brndnhghs/grillmaster-command-center.git
cd grillmaster-command-center

# Create the venv the services expect (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Install the pinned dependency set
pip install -r requirements.txt
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

**Optional extras** (commented out in `requirements.txt`; each is imported lazily inside the method that needs it, so the server runs without them — only that method fails until installed): `matplotlib` (colormaps), `scikit-image` (fractal resize), `pyfiglet` (ASCII art), `qrcode` (#09 QR Code), `moderngl` (#82 GPU Shaders), `torch` + `diffusers` (#21 Stable Diffusion 1.5, ~2 GB+).

> **Chord Bot** has its own `chord_bot/pyproject.toml`, but it shares the same `.venv` — no separate install is required once `requirements.txt` is satisfied.

## First Run

The simplest path is the Dashboard, which launches and monitors both services:

```bash
source .venv/bin/activate
python -m dashboard --autostart
# → http://127.0.0.1:7870
```

Open `http://127.0.0.1:7870` in a browser. The dashboard shows Launch/Stop controls for each service and an embedded UI switcher.

**Alternative — direct launcher script:**

```bash
bash scripts/grillmaster-launcher.sh
```

> ⚠️ `scripts/grillmaster-launcher.sh` hard-codes a machine-specific venv path (`/Users/admin/Documents/GitHub/hermes-agent/venv/bin/python`). It works on this machine but is **not portable** — prefer `python -m dashboard` elsewhere.

## Service Ports

| Service | Port | Launched by |
|----------|-------|-------------|
| Image Pipeline (server) | `7860` | Dashboard / launcher |
| Chord Bot (server) | `7861` | Dashboard / launcher |
| 3D Sidecar (Node.js) | `7862` | Dashboard |
| Dashboard (control panel) | `7870` | you |

You can also run a single service directly:

```bash
python -m image_pipeline.server --port 7860
python -m chord_bot.server     --port 7861
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

### Chord progression
Open Chord Bot (`:7861`). Place horizontal nodes (Tonic, Function, Cadence…) left-to-right, add vertical augmenters, then export as MIDI, text chart, or JSON.

## Configuration

- **`GRILLMASTER_API_TOKEN`** — when set, the server requires this token on protected endpoints (Node Doctor apply/undo, Node Tester batch-apply). The UI reads it from `localStorage['api-token']` and attaches it as `X-Api-Token` on every request. No-op when unset (local/dev).
- **`THREEJS_SIDECAR_URL`** — default `http://127.0.0.1:7862`. The server proxies 3D-node graph renders to this Node.js sidecar. Override to point at a remote sidecar.
- **`data/logs/`** — service stdout/stderr are written here (e.g. `data/logs/pipeline.log`). Useful when a launch reports "failed".

## Where to Go Next

- Architecture: [architecture.md](architecture.md)
- Module reference: [README.md#module-map](README.md#module-map)
- HTTP API index: [api.md](api.md)
- Chord Bot deep-dive: [modules/chord-bot.md](modules/chord-bot.md)
