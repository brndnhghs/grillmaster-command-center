"""Synchronous Hermes runner — the sole LLM backend (same as Node Doctor).

Mirrors the subprocess contract in `server.py::_nd_stream` and `nd_runner.py`
but returns the full completion as a single string, so the tuning package can
call it directly (builder, learn) and be unit-tested by monkeypatching
`run_hermes`. Path resolution matches `server.py:2518` exactly, so both agree.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# nd_runner.py lives next to this package's parent (image_pipeline/nd_runner.py).
_ND_RUNNER = Path(__file__).resolve().parent.parent / "nd_runner.py"

HERMES_AGENT_DIR = Path(
    os.environ.get("HERMES_AGENT_DIR", str(Path.home() / ".hermes" / "hermes-agent"))
)
HERMES_PY = Path(
    os.environ.get("HERMES_PYTHON", str(HERMES_AGENT_DIR / "venv" / "bin" / "python"))
)


class HermesUnavailable(RuntimeError):
    """Raised when the Hermes backend interpreter cannot be found."""


def hermes_available() -> bool:
    return HERMES_PY.exists() and _ND_RUNNER.exists()


def unavailable_message() -> str:
    return (
        f"⚠ Hermes backend not found at {HERMES_PY}. "
        f"Set HERMES_AGENT_DIR (or HERMES_PYTHON) and restart."
    )


def run_hermes(system: str, messages: list[dict], timeout: int = 180) -> str:
    """Run one Hermes completion. Returns the concatenated text.

    `messages` is a list of {role, content}; the last user message is the turn.
    Raises HermesUnavailable if the backend is missing, RuntimeError on a
    runner-level error.
    """
    if not hermes_available():
        raise HermesUnavailable(unavailable_message())

    stdin_bytes = json.dumps(
        {"system_prompt": system, "messages": messages}
    ).encode()

    proc = subprocess.run(
        [str(HERMES_PY), str(_ND_RUNNER)],
        input=stdin_bytes,
        capture_output=True,
        timeout=timeout,
    )

    if proc.returncode != 0 and not proc.stdout.strip():
        err = proc.stderr.decode()[:500] if proc.stderr else "no output"
        raise RuntimeError(f"Hermes runner failed: {err}")

    chunks: list[str] = []
    for raw in proc.stdout.decode().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "text" in d:
            chunks.append(d["text"])
        elif "error" in d:
            raise RuntimeError(d["error"])
    return "".join(chunks)
