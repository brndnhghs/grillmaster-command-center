#!/usr/bin/env bash
# image_wiring_cron.sh — cron entrypoint for image-input wiring validation.
# Watchdog pattern: silent when every graph is clean; on any ERROR-level
# finding, emit the full report (delivered verbatim to the user) and exit 2
# so the cron run is flagged. Uses the project's uv venv without shadowing
# numpy via the harness PYTHONPATH (which trips a consent gate).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

unset PYTHONPATH

# Capture stdout AND exit code (must disable -e so a non-zero return doesn't abort).
set +e
OUT="$(.venv/bin/python tools/validate_image_wiring.py "$@" 2>&1)"
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
    # Clean — say nothing so the cron stays silent.
    exit 0
fi

# Non-zero (errors, or fatal): surface the report.
echo "$OUT"
exit 2
