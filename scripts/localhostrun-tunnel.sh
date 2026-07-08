#!/bin/bash
# Grillmaster — localhost.run tunnels (self-healing via watchdog).
#
# Two modes:
#   FREE (rotating):   no account file -> uses nokey@localhost.run, gets a
#                      fresh *.lhr.life subdomain every restart (bookmark dies).
#   ACCOUNT (stable):  if data/lr-account.txt exists and is non-empty, its
#                      contents are used as the localhost.run account name.
#                      Connecting with the registered SSH key yields a
#                      PERSISTENT subdomain (e.g. <account>.lhrtunnel.link)
#                      that does NOT change between restarts.
#
# To enable stable mode (one-time, ~30s):
#   1. create a free account at https://admin.localhost.run/
#   2. add your SSH public key  (~/.ssh/localhost_run.pub) to the account
#   3. echo "<your-account-name>" > data/lr-account.txt
# The key is already generated at ~/.ssh/localhost_run(.pub).
#
# Writes data/tunnel-info.json (consumed by the dashboard, injected
# server-side into the iframe URLs) and data/remote-url.txt.
set -uo pipefail

REPO="/Users/admin/Documents/GitHub/grillmaster-command-center"
DATA_DIR="$REPO/data"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"
SSH_KEY="$HOME/.ssh/localhost_run"

port_for(){ case "$1" in dashboard) echo 7870;; pipeline) echo 7860;; chord) echo 7861;; esac; }

# Account mode?
ACCOUNT=""
if [ -f "$DATA_DIR/lr-account.txt" ]; then
  ACCOUNT="$(tr -d '[:space:]' < "$DATA_DIR/lr-account.txt")"
fi

if [ -n "$ACCOUNT" ]; then
  LH_USER="$ACCOUNT@localhost.run"
  echo "[tunnel] ACCOUNT mode -> stable subdomain for account '$ACCOUNT'"
else
  LH_USER="nokey@localhost.run"
  echo "[tunnel] FREE mode -> rotating *.lhr.life subdomain"
fi

ssh_opts_base=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -i "$SSH_KEY")

start_one(){
  local name="$1" port="$2"
  local log="$LOG_DIR/${name}-localhostrun.log"
  : > "$log"
  ssh "${ssh_opts_base[@]}" -R 80:localhost:$port "$LH_USER" \
    > "$log" 2>&1 &
  echo $! > "$LOG_DIR/${name}.pid"
}

# Kill any existing
for n in dashboard pipeline chord; do
  [ -f "$LOG_DIR/${n}.pid" ] && kill -9 "$(cat "$LOG_DIR/${n}.pid")" 2>/dev/null || true
done
pkill -f "localhost.run" 2>/dev/null || true
sleep 1

for n in dashboard pipeline chord; do
  start_one "$n" "$(port_for "$n")"
done

# Wait for URLs to appear, then write tunnel-info.json + remote-url.txt
sleep 3
DASH_URL="UNKNOWN"; PIPE_URL="UNKNOWN"; CHORD_URL="UNKNOWN"
for n in dashboard pipeline chord; do
  local_log="$LOG_DIR/${n}-localhostrun.log"
  u=""
  for i in $(seq 1 30); do
    u=$(grep -oE 'https://[a-zA-Z0-9.-]+\.(lhr\.life|lhrtunnel\.link|lhr\.rocks)' "$local_log" 2>/dev/null | head -1)
    [ -n "$u" ] && break
    sleep 1
  done
  echo "$n -> ${u:-UNKNOWN}"
  case "$n" in
    dashboard) DASH_URL="${u:-UNKNOWN}";;
    pipeline)  PIPE_URL="${u:-UNKNOWN}";;
    chord)     CHORD_URL="${u:-UNKNOWN}";;
  esac
done

cat > "$DATA_DIR/tunnel-info.json" <<EOF
{
  "dashboard": {"url": "$DASH_URL", "local": "http://127.0.0.1:7870"},
  "pipeline":  {"url": "$PIPE_URL",  "local": "http://127.0.0.1:7860"},
  "chord":     {"url": "$CHORD_URL", "local": "http://127.0.0.1:7861"}
}
EOF
echo "$DASH_URL" > "$DATA_DIR/remote-url.txt"
echo "WROTE tunnel-info.json + remote-url.txt"
echo "DASHBOARD: $DASH_URL"
