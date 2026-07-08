#!/bin/bash
# Grillmaster Command Center — Dashboard launcher.
# Starts the unified control panel (port 7870). From there you can launch the
# Image Pipeline (7860) and Chord Bot (7861) with one click.
# Run with: bash scripts/dashboard.sh   (optionally: --autostart)
set -euo pipefail

REPO="/Users/admin/Documents/GitHub/grillmaster-command-center"
VENV_PY="$REPO/.venv/bin/python"
DASH_PORT=7870

mkdir -p "$REPO/data/logs"

# Kill any stale dashboard on our port
lsof -ti:"$DASH_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

echo "Starting Grillmaster Command Center dashboard on :$DASH_PORT ..."
PYTHONPATH="$REPO" nohup "$VENV_PY" -m dashboard "$@" > "$REPO/data/logs/dashboard.log" 2>&1 &

for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:$DASH_PORT/" > /dev/null 2>&1; then
    echo "✓ Dashboard ready → http://127.0.0.1:$DASH_PORT"
    break
  fi
  if [ "$i" -eq 15 ]; then echo "✗ Dashboard failed to start (see data/logs/dashboard.log)" >&2; exit 1; fi
  sleep 1
done

echo ""
echo "  Open:  http://127.0.0.1:$DASH_PORT"
echo "  Then click 'Launch Both' to boot the Image Pipeline + Chord Bot."
