#!/bin/bash
# Chord Bot Launcher — starts the server (local only).
# Tunnelling was removed and will be rebuilt later.
# Run with: nohup ./scripts/chord-bot-launcher.sh &
set -euo pipefail

PORT=7861
DATA_DIR="/Users/admin/Documents/GitHub/grillmaster-command-center/data"
CHORD_DIR="/Users/admin/Documents/GitHub/grillmaster-command-center/chord_bot"
VENV_PYTHON="/Users/admin/Documents/GitHub/hermes-agent/venv/bin/python"
# Clear PYTHONPATH/_VIRTUAL_ENV so this venv isn't shadowed by the Hermes
# shell's inherited py3.11 site-packages (breaks numpy under py3.12).
ENV_PREFIX="env -u PYTHONPATH -u _VIRTUAL_ENV PYTHONPATH=$CHORD_DIR"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

# Kill any stale processes on our port
lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# Start Chord Bot server (no trap — runs independently)
echo "Starting Chord Bot server..."
$ENV_PREFIX nohup "$VENV_PYTHON" -m chord_bot.server --port "$PORT" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "✓ Server ready at http://127.0.0.1:$PORT (PID $SERVER_PID)"
    break
  fi
  if [ "$i" -eq 15 ]; then
    echo "✗ Server failed to start" >&2
    exit 1
  fi
  sleep 1
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  Chord Bot is LIVE"
echo "═══════════════════════════════════════════════"
echo "  Local:   http://127.0.0.1:$PORT"
echo "═══════════════════════════════════════════════"
echo ""
echo "Stop with: kill $SERVER_PID"
