#!/bin/bash
# Start the Chord Bot server standalone (without the image pipeline).
# When the image pipeline is running, Chord Bot is already served at /chordbot/.
# Use this script only for isolated Chord Bot development.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${CHORDBOT_PORT:-7861}"
cd "$REPO"

echo "Chord Bot  →  http://127.0.0.1:${PORT}"
exec .venv/bin/uvicorn chord_bot.server:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --no-access-log
