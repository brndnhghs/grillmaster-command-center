#!/bin/bash
# Grillmaster tunnel watchdog — keeps localhost.run tunnels alive.
# Run via Hermes cron every 10 min. Re-establishes tunnels if any are down
# and rewrites data/tunnel-info.json so the dashboard iframes stay correct.
set -u
REPO="/Users/admin/Documents/GitHub/grillmaster-command-center"
DATA_DIR="$REPO/data"
LOG="$DATA_DIR/logs/watchdog.log"
mkdir -p "$DATA_DIR/logs"

dash_url=$(python3 -c "import json,sys; print(json.load(open('$DATA_DIR/tunnel-info.json')).get('dashboard',{}).get('url',''))" 2>/dev/null || true)

alive=1
if [ -n "$dash_url" ]; then
  code=$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 8 "$dash_url/" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && alive=1 || alive=0
else
  alive=0
fi

# Also confirm at least one localhost.run ssh proc exists
lr_procs=$(pgrep -f "localhost.run" | wc -l | tr -d ' ')
if [ "$lr_procs" -eq 0 ]; then alive=0; fi

if [ "$alive" -eq 1 ]; then
  echo "$(date) OK — tunnels up ($dash_url, $lr_procs ssh procs)" >> "$LOG"
  exit 0
fi

echo "$(date) TUNNELS DOWN — restarting via localhostrun-tunnel.sh" >> "$LOG"
bash "$REPO/scripts/localhostrun-tunnel.sh" >> "$LOG" 2>&1
echo "$(date) restart done" >> "$LOG"
