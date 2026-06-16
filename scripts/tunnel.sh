#!/bin/bash
set -euo pipefail
PORT="${GRILLMASTER_TUNNEL_PORT:-8766}"
DATA_DIR="/Users/admin/Documents/GitHub/grillmaster-command-center/data"
URL_FILE="$DATA_DIR/remote-url.txt"
LOG_FILE="$DATA_DIR/localhost-run.log"
mkdir -p "$DATA_DIR"
: > "$LOG_FILE"
rm -f "$URL_FILE"
/usr/bin/ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o LogLevel=ERROR -R 80:localhost:${PORT} nokey@localhost.run 2>&1 | /usr/bin/tee -a "$LOG_FILE" | /usr/bin/awk -v urlfile="$URL_FILE" '
  /tunneled with tls termination, https:\/\// {
    if (match($0, /https:\/\/[A-Za-z0-9.-]+/)) {
      print substr($0, RSTART, RLENGTH) > urlfile
      close(urlfile)
    }
  }
  { print; fflush() }
'
