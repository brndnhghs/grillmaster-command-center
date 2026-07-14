#!/usr/bin/env bash
# cron_image_input.sh — repo entrypoint for the image-input contract audit.
#
# Resolves its REAL (symlink-free) location so it always runs from the repo
# root, whether invoked as:
#   bash image_pipeline/tools/cron_image_input.sh --report        (from repo root)
#   bash ~/.hermes/scripts/image_pipeline/tools/cron_image_input.sh --report  (cron)
# Then execs the Python auditor inside the project's uv venv (unset PYTHONPATH
# so the harness numpy is not shadowed — that trips a consent gate).
#
# Usage:
#   --report   read-only audit + report to stdout (writes report artifacts only)
#   --apply    SAFE-class fixes only; ABORTS if the git tree is dirty
set -euo pipefail

# Resolve through any symlinks to the real source file.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO"

unset PYTHONPATH

exec "$REPO/.venv/bin/python" "$REPO/tools/cron_image_input.py" "$@"
