"""Grillmaster Command Center — Dashboard backend.

A single FastAPI app (port 7870) that launches, monitors, and stops the two
Grillmaster services — the Image Pipeline (7860) and the Chord Bot (7861) —
and presents a unified control panel + embedded UI switcher.

Run:
    python -m dashboard            # launches servers on demand via the UI
    python -m dashboard --autostart  # also boots both services at startup

The dashboard does NOT itself render either app; it spawns each service as a
child process (using the repo .venv) and proxies their status via /health.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
UI_DIR = Path(__file__).resolve().parent / "ui"

# Use the repo .venv (Py 3.12) which has both chord_bot and image_pipeline.
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

PIPELINE_PORT = 7860
CHORD_PORT = 7861
DASHBOARD_PORT = 7870

# ── Process registry ──────────────────────────────────────────────────────────
_PROCS: dict[str, subprocess.Popen] = {}


def _spawn(name: str, module: str, port: int) -> subprocess.Popen:
    """Spawn a service as a backgrounded subprocess using the repo venv."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    # chord_bot module lives under chord_bot/ (repo root on PYTHONPATH handles it)
    proc = subprocess.Popen(
        [str(VENV_PYTHON), "-m", module, "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=(DATA_DIR / "logs" / f"{name}.log").open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _PROCS[name] = proc
    return proc


def _stop(name: str) -> None:
    proc = _PROCS.get(name)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    _PROCS.pop(name, None)


def _is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _is_healthy(port: int, timeout: float = 1.0) -> bool:
    """True only if /health actually answers — a listening socket is not enough.
    A wedged server keeps its socket in LISTEN while answering nothing."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _reclaim_port(port: int) -> bool:
    """Kill whatever is listening on `port`. Returns True if something was killed.

    Without this a leftover instance (typically one that hung and stopped
    answering) keeps the port, every relaunch dies with 'address already in
    use', and the UI reports 'starting' forever.
    """
    try:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout.split()
    except Exception:
        return False
    killed = False
    for raw in out:
        try:
            pid = int(raw)
        except ValueError:
            continue
        try:
            # Kill the whole group like _stop does — killing only the listener
            # would orphan any children it spawned.
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            killed = True
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
                killed = True
            except (ProcessLookupError, PermissionError):
                pass
    if killed:
        time.sleep(1.0)
    return killed


def _wait_ready(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_healthy(port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _begin_start(name: str, cfg: dict) -> bool:
    """Spawn one service, reclaiming a stale port first.

    Returns False if it was already up and healthy (nothing spawned).
    """
    port = cfg["port"]
    proc = _PROCS.get(name)
    if proc and proc.poll() is None and _is_healthy(port):
        return False
    if _is_port_open(port):
        if _is_healthy(port):
            return False
        _reclaim_port(port)   # listening but wedged/orphaned — take the port back
    _stop(name)
    if cfg.get("node"):
        _spawn_node(name, cfg["node"], port)
    else:
        _spawn(name, cfg["module"], port)
    return True


def _finish_start(name: str, port: int) -> str:
    """Wait for a just-spawned service and report what actually happened."""
    if _wait_ready(port):
        return "running"
    p = _PROCS.get(name)
    if p is not None and p.poll() is not None:
        return f"failed (exited {p.returncode}, see data/logs/{name}.log)"
    return f"failed (no response on :{port}, see data/logs/{name}.log)"


def service_status(name: str, port: int) -> dict:
    """Return the live status of a service (process + /health if reachable)."""
    proc = _PROCS.get(name)
    running = proc is not None and proc.poll() is None
    info = {
        "name": name,
        "port": port,
        "pid": proc.pid if running else None,
        "process_alive": running,
        "port_open": _is_port_open(port),
    }
    if info["port_open"]:
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as r:
                info["health"] = r.status
        except Exception:
            info["health"] = None
    else:
        info["health"] = None
    info["url"] = f"http://127.0.0.1:{port}"
    return info


NODEJS = REPO_ROOT / "image_pipeline" / "3d" / "threejs-sidecar.mjs"
THREEJS_PORT = 7862

SERVICES = {
    "pipeline": {"module": "image_pipeline.server", "port": PIPELINE_PORT},
    "chord": {"module": "chord_bot.server", "port": CHORD_PORT},
    "3d": {"module": None, "port": THREEJS_PORT, "node": NODEJS},
}


def _spawn_node(name: str, script: Path, port: int) -> subprocess.Popen:
    """Spawn a Node.js sidecar as a backgrounded subprocess."""
    env = dict(os.environ)
    env["THREEJS_PORT"] = str(port)
    proc = subprocess.Popen(
        ["node", str(script)],
        cwd=str(script.parent),
        env=env,
        stdout=(DATA_DIR / "logs" / f"{name}.log").open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _PROCS[name] = proc
    return proc


def launch_all() -> dict:
    # Spawn everything first, then wait — waiting per service in turn would
    # stack one readiness timeout on top of the next.
    results: dict[str, str] = {}
    starting: list[str] = []
    for name, cfg in SERVICES.items():
        if _begin_start(name, cfg):
            starting.append(name)
        else:
            results[name] = "already running"
    for name in starting:
        results[name] = _finish_start(name, SERVICES[name]["port"])
    return results


def stop_all() -> dict:
    for name in list(_PROCS):
        _stop(name)
    # Same reason as api_stop_one: untracked leftovers must lose the port too.
    time.sleep(0.5)
    for cfg in SERVICES.values():
        if _is_port_open(cfg["port"]):
            _reclaim_port(cfg["port"])
    return {name: "stopped" for name in SERVICES}


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Grillmaster Command Center", docs_url=None, redoc_url=None)


@app.on_event("startup")
def _startup() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse({
        "dashboard_port": DASHBOARD_PORT,
        "services": {name: service_status(name, cfg["port"]) for name, cfg in SERVICES.items()},
    })


@app.post("/api/launch")
def api_launch() -> JSONResponse:
    return JSONResponse(launch_all())


@app.post("/api/stop")
def api_stop() -> JSONResponse:
    return JSONResponse(stop_all())


@app.post("/api/launch/{name}")
def api_launch_one(name: str) -> JSONResponse:
    cfg = SERVICES.get(name)
    if cfg is None:
        return JSONResponse({"error": f"unknown service {name}"}, status_code=404)
    if not _begin_start(name, cfg):
        return JSONResponse({name: "already running"})
    status = _finish_start(name, cfg["port"])
    return JSONResponse({name: status},
                        status_code=200 if status == "running" else 500)


@app.post("/api/stop/{name}")
def api_stop_one(name: str) -> JSONResponse:
    cfg = SERVICES.get(name)
    if cfg is None:
        return JSONResponse({"error": f"unknown service {name}"}, status_code=404)
    _stop(name)
    # The dashboard forgets _PROCS across its own restarts, so also clear
    # anything still holding the port — otherwise Stop looks like a no-op.
    time.sleep(0.5)
    if _is_port_open(cfg["port"]):
        _reclaim_port(cfg["port"])
    return JSONResponse({name: "stopped"})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((UI_DIR / "index.html").read_text())


# Serve the dashboard's own static assets (css/js) if present.
if (UI_DIR / "app.js").exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


def run(host: str = "127.0.0.1", port: int = DASHBOARD_PORT, autostart: bool = False) -> None:
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Grillmaster Command Center dashboard")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--autostart", action="store_true", help="Launch both services on boot")
    args = parser.parse_args()
    if args.autostart or autostart:
        launch_all()
    print(f"  Grillmaster Command Center  →  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
