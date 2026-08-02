#!/bin/bash
# Grillmaster Launcher — starts the Image Pipeline (local only).
# Tunnelling was removed and will be rebuilt later.
# Run with: bash scripts/grillmaster-launcher.sh
set -euo pipefail

# Repo root derived from this script's own location — portable across machines
# (macOS, Linux, Windows git-bash) and checkouts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
PIPELINE_PORT=7860
DATA_DIR="$REPO/data"
PIPELINE_DIR="$REPO/image_pipeline"
VENV_PYTHON="$REPO/.venv/bin/python"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

# Kill any stale process on our port
lsof -ti:$PIPELINE_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# ── Start Image Pipeline server ──
echo "Starting Image Pipeline server..."
# Clear PYTHONPATH/_VIRTUAL_ENV so the agent venv isn't shadowed by the
# Hermes shell's inherited py3.11 site-packages (which breaks numpy import).
env -u PYTHONPATH -u _VIRTUAL_ENV PYTHONPATH="$REPO" nohup "$VENV_PYTHON" -m image_pipeline.server --port "$PIPELINE_PORT" > "$LOG_DIR/pipeline-server.log" 2>&1 &
PIPE_PID=$!

for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:$PIPELINE_PORT/health" > /dev/null 2>&1; then
    echo "✓ Image Pipeline ready at http://127.0.0.1:$PIPELINE_PORT (PID $PIPE_PID)"
    break
  fi
  if [ "$i" -eq 15 ]; then echo "✗ Image Pipeline failed to start" >&2; exit 1; fi
  sleep 1
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  Grillmaster is LIVE"
echo "═══════════════════════════════════════════════"
echo "  Image Pipeline:  http://127.0.0.1:$PIPELINE_PORT"
echo "═══════════════════════════════════════════════"
echo ""
echo "Stop with: kill $PIPE_PID"
