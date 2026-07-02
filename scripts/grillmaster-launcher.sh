#!/bin/bash
# Grillmaster Launcher — starts Chord Bot + Image Pipeline + Cloudflare tunnels
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
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

# ── 1. Start Chord Bot server ──
echo "Starting Chord Bot server..."
PYTHONPATH="/Users/admin/Documents/GitHub/grillmaster-command-center" nohup "$VENV_PYTHON" -m chord_bot.server --port "$CHORD_PORT" > "$LOG_DIR/chord-server.log" 2>&1 &
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
PYTHONPATH="/Users/admin/Documents/GitHub/grillmaster-command-center" nohup "$VENV_PYTHON" -m image_pipeline.server --port "$PIPELINE_PORT" > "$LOG_DIR/pipeline-server.log" 2>&1 &
PIPE_PID=$!

for i in $(seq 1 15); do
  if curl -sf "http://127.0.0.1:$PIPELINE_PORT/health" > /dev/null 2>&1; then
    echo "✓ Image Pipeline ready at http://127.0.0.1:$PIPELINE_PORT (PID $PIPE_PID)"
    break
  fi
  if [ "$i" -eq 15 ]; then echo "✗ Image Pipeline failed to start" >&2; exit 1; fi
  sleep 1
done

# ── 3. Start Cloudflare tunnels ──
echo "Starting Cloudflare tunnels..."

# Chord Bot tunnel
nohup cloudflared tunnel --url "http://localhost:$CHORD_PORT" > "$LOG_DIR/chord-tunnel.log" 2>&1 &
CHORD_TUNNEL_PID=$!

# Image Pipeline tunnel
nohup cloudflared tunnel --url "http://localhost:$PIPELINE_PORT" > "$LOG_DIR/pipeline-tunnel.log" 2>&1 &
PIPE_TUNNEL_PID=$!

# Wait for both tunnel URLs
CHORD_URL=""
PIPE_URL=""
for i in $(seq 1 45); do
  if [ -z "$CHORD_URL" ]; then
    CHORD_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/chord-tunnel.log" 2>/dev/null | tail -1)
  fi
  if [ -z "$PIPE_URL" ]; then
    PIPE_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/pipeline-tunnel.log" 2>/dev/null | tail -1)
  fi
  [ -n "$CHORD_URL" ] && [ -n "$PIPE_URL" ] && break
  sleep 1
done

echo "$CHORD_URL" > "$DATA_DIR/tunnel-url.txt"

# Write tunnel info for the UI
cat > "$DATA_DIR/tunnel-info.json" <<EOF
{
  "chord": {"url": "$CHORD_URL", "local": "http://127.0.0.1:$CHORD_PORT"},
  "pipeline": {"url": "$PIPE_URL", "local": "http://127.0.0.1:$PIPELINE_PORT"}
}
EOF

echo ""
echo "═══════════════════════════════════════════════"
echo "  Grillmaster is LIVE"
echo "═══════════════════════════════════════════════"
echo "  Chord Bot:"
echo "    Local:   http://127.0.0.1:$CHORD_PORT"
echo "    Public:  $CHORD_URL"
echo ""
echo "  Image Pipeline:"
echo "    Local:   http://127.0.0.1:$PIPELINE_PORT"
echo "    Public:  $PIPE_URL"
echo "═══════════════════════════════════════════════"
echo ""
echo "Stop with: kill $CHORD_PID $PIPE_PID $CHORD_TUNNEL_PID $PIPE_TUNNEL_PID"
