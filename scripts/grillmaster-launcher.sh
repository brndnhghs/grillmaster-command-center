#!/bin/bash
# Grillmaster Launcher — starts Chord Bot + Image Pipeline (local only).
# Tunnelling was removed and will be rebuilt later.
# Run with: bash scripts/grillmaster-launcher.sh
set -euo pipefail

CHORD_PORT=7861
PIPELINE_PORT=7860
DATA_DIR="/Users/admin/Documents/GitHub/grillmaster-command-center/data"
CHORD_DIR="/Users/admin/Documents/GitHub/grillmaster-command-center/chord_bot"
PIPELINE_DIR="/Users/admin/Documents/GitHub/grillmaster-command-center/image_pipeline"
VENV_PYTHON="/Users/admin/Documents/GitHub/hermes-agent/venv/bin/python"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

# Kill any stale processes on our ports
lsof -ti:$CHORD_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:$PIPELINE_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# ── 1. Start Chord Bot server ──
echo "Starting Chord Bot server..."
# Clear PYTHONPATH/_VIRTUAL_ENV so the agent venv isn't shadowed by the
# Hermes shell's inherited py3.11 site-packages (which breaks numpy import).
env -u PYTHONPATH -u _VIRTUAL_ENV PYTHONPATH="/Users/admin/Documents/GitHub/grillmaster-command-center" nohup "$VENV_PYTHON" -m chord_bot.server --port "$CHORD_PORT" > "$LOG_DIR/chord-server.log" 2>&1 &
CHORD_PID=$!

for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:$CHORD_PORT/health" > /dev/null 2>&1; then
    echo "✓ Chord Bot ready at http://127.0.0.1:$CHORD_PORT (PID $CHORD_PID)"
    break
  fi
  if [ "$i" -eq 15 ]; then echo "✗ Chord Bot failed to start" >&2; exit 1; fi
  sleep 1
done

# ── 2. Start Image Pipeline server ──
echo "Starting Image Pipeline server..."
env -u PYTHONPATH -u _VIRTUAL_ENV PYTHONPATH="/Users/admin/Documents/GitHub/grillmaster-command-center" nohup "$VENV_PYTHON" -m image_pipeline.server --port "$PIPELINE_PORT" > "$LOG_DIR/pipeline-server.log" 2>&1 &
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
echo "  Chord Bot:       http://127.0.0.1:$CHORD_PORT"
echo "  Image Pipeline:  http://127.0.0.1:$PIPELINE_PORT"
echo "═══════════════════════════════════════════════"
echo ""
echo "Stop with: kill $CHORD_PID $PIPE_PID"
