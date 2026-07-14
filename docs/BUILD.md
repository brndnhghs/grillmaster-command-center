# Build Guide — Grillmaster Command Center

> Generated: 2026-07-13 · Phase 5

---

## Prerequisites

- Python 3.10+
- uv (preferred) or pip
- ffmpeg (for MP4 encoding)
- Optional: Blender (for 3D rendering), ModernGL (for GPU shaders)

## Setup

```bash
# Clone
git clone https://github.com/brndnhghs/grillmaster-command-center.git
cd grillmaster-command-center

# Create virtual environment
uv venv .venv

# Install dependencies
uv pip install -r requirements.txt --python .venv/bin/python

# Install pre-commit hook (optional)
pre-commit install
```

## Run

### Option A — Unified Dashboard (recommended)
```bash
bash scripts/dashboard.sh              # http://localhost:7870
bash scripts/dashboard.sh --autostart  # boots both services
```

### Option B — Individual Services
```bash
# Image Pipeline (port 7860)
.venv/bin/python -m image_pipeline.server

# Chord Bot (port 7861)
.venv/bin/python -m chord_bot.server
```

## Tunneling

```bash
# Expose the server to the internet via pyngrok
bash scripts/tunnel.sh
```

## Smoke Test

```bash
# Verify all methods register without errors
uv run python -c "from image_pipeline.server import app"
```

## Pre-flight Checklist (before committing method changes)

1. Check method ID is unique: `uv run python tools/next_id.py`
2. Run method audit: `uv run python tools/audit_methods.py --fail-on-violations`
3. Run fast tests: `uv run pytest -q`
4. Server import test: `uv run python -c "from image_pipeline.server import app"`
5. If changing core: update DESIGN.md