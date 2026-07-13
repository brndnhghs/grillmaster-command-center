#!/usr/bin/env bash
# validate_image_wiring.sh — run the image-input wiring validator inside the
# Grillmaster uv venv. The project's numpy must NOT be shadowed by the harness
# PYTHONPATH (it trips a consent gate), so we unset it and use .venv explicitly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

unset PYTHONPATH
exec .venv/bin/python tools/validate_image_wiring.py "$@"
