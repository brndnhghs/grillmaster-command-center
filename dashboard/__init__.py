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
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
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
    results = {}
    for name, cfg in SERVICES.items():
        if _PROCS.get(name) and _PROCS[name].poll() is None:
            results[name] = "already running"
            continue
        if cfg.get("node"):
            _spawn_node(name, cfg["node"], cfg["port"])
        else:
            _spawn(name, cfg["module"], cfg["port"])
        results[name] = "starting"
    return results


def stop_all() -> dict:
    for name in list(_PROCS):
        _stop(name)
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


@app.get("/api/tunnel-info")
def api_tunnel_info() -> JSONResponse:
    """Public backend URLs for embedding through a tunnel.

    When the dashboard is reached via a public tunnel, the embedded iframes
    and 'Open' links must point at the backends' PUBLIC urls (not 127.0.0.1,
    which is the visitor's own localhost). Sourced from data/tunnel-info.json,
    which the launch/tunnel scripts write. Missing keys fall back to
    127.0.0.1 in the UI.
    """
    info: dict = {}
    p = DATA_DIR / "tunnel-info.json"
    if p.exists():
        try:
            info = json.loads(p.read_text())
        except Exception:
            info = {}
    return JSONResponse(info)


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
    if _PROCS.get(name) and _PROCS[name].poll() is None:
        return JSONResponse({name: "already running"})
    _spawn(name, cfg["module"], cfg["port"])
    return JSONResponse({name: "starting"})


@app.post("/api/stop/{name}")
def api_stop_one(name: str) -> JSONResponse:
    cfg = SERVICES.get(name)
    if cfg is None:
        return JSONResponse({"error": f"unknown service {name}"}, status_code=404)
    _stop(name)
    return JSONResponse({name: "stopped"})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (UI_DIR / "index.html").read_text()
    # Inject the live public backend URLs server-side so the embedded iframes
    # work through a tunnel — no client-side fetch race, and it survives
    # subdomain rotation (tunnel-info.json is re-read on every request).
    tunnel = {"pipeline": None, "chord": None}
    dash_url = None
    p = DATA_DIR / "tunnel-info.json"
    if p.exists():
        try:
            info = json.loads(p.read_text())
            for k in ("pipeline", "chord"):
                if isinstance(info.get(k), dict) and info[k].get("url"):
                    tunnel[k] = {"url": info[k]["url"]}
            if isinstance(info.get("dashboard"), dict) and info["dashboard"].get("url"):
                dash_url = info["dashboard"]["url"]
        except Exception:
            pass
    html = html.replace(
        "const TUNNEL = { pipeline: null, chord: null };",
        "const TUNNEL = " + json.dumps(tunnel) + ";",
        1,
    )
    if dash_url:
        html = html.replace(
            '<span class="pill" id="dash-pill">dashboard :7870</span>',
            f'<span class="pill" id="dash-pill" title="{dash_url}">🌐 {dash_url.replace("https://", "")}</span>',
            1,
        )
    return HTMLResponse(html)


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
