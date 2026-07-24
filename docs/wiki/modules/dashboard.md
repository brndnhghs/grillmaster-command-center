# Module: `dashboard`

## Purpose
Process supervisor and unified control panel for the whole Grillmaster stack. A small FastAPI app (port **7870**) that spawns, monitors, and stops the two main services — the Image Pipeline (`:7860`) and the Chord Bot (`:7861`) — plus the three.js sidecar (`:7862`) — and serves a single-page UI that switches between them. It does NOT render either app itself; it launches each as a child process using the repo `.venv` and proxies status via `/health` checks.

## Responsibilities
- Spawn each service as a backgrounded subprocess (repo `.venv` Python, `PYTHONPATH` set to repo root)
- Stop services (kills the whole process group, not just the listener)
- Reclaim a stale/orphaned port before relaunching (a hung server keeps its socket open)
- Health-check via `/health` (a listening socket alone is not enough — a wedged server answers nothing)
- Expose launch/stop/status endpoints for the UI
- Serve the dashboard SPA (`ui/index.html`)

## Key Functions

### `_spawn(name, module, port) -> subprocess.Popen`
Launches `python -m <module> --port <port>` under the repo venv, redirecting stdout/stderr to `data/logs/<name>.log`, in a new session so it survives the parent.

### `_spawn_node(name, script, port)`
Spawns the Node.js three.js sidecar (`image_pipeline/3d/threejs-sidecar.mjs`) via `node`.

### `_stop(name)`
Kills the service's process group (`os.killpg`) so any children it spawned die too.

### `_reclaim_port(port) -> bool`
Uses `lsof` to find whatever holds `tcp:<port>` in LISTEN and `SIGKILL`s it (group-first, then PID). Returns True if something was killed. Prevents "address already in use" on relaunch.

### `_is_healthy(port) -> bool`
Opens `http://127.0.0.1:<port>/health` and requires a 200 — distinguishes "port open but dead" from "actually serving".

### `launch_all() / stop_all()`
Iterate `SERVICES` (pipeline, chord, 3d). `launch_all` spawns everything first, then waits — avoids stacking readiness timeouts.

### `service_status(name, port) -> dict`
Returns process-alive, port-open, and health status for one service.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Dashboard port + per-service status (process, port, health, url) |
| POST | `/api/launch` | Launch all three services |
| POST | `/api/stop` | Stop all services |
| POST | `/api/launch/{name}` | Launch one service (`pipeline` / `chord` / `3d`) |
| POST | `/api/stop/{name}` | Stop one service (also reclaims its port) |
| GET | `/` | Serve the dashboard SPA |

## Launch
```bash
python -m dashboard            # launches services on demand via the UI
python -m dashboard --autostart  # also boots both services at startup
```

## Dependencies
- `fastapi`, `uvicorn`
- stdlib: `subprocess`, `signal`, `socket`, `urllib.request`
- The repo `.venv` Python (3.12) — has both `chord_bot` and `image_pipeline` on path

## Key Design Notes
- **Forgets `_PROCS` across its own restarts** — so after a dashboard restart, `api_stop_one` also calls `_reclaim_port` to free any orphaned listener.
- **Port reclaim before spawn** — a leftover wedged instance would otherwise block every relaunch.
- **3D sidecar** is a Node.js process (`threejs-sidecar.mjs`), not a Python module; spawned via `_spawn_node`.

## Source
[`dashboard/__init__.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/3e085d44fccca63896b5f6543aaa54ab4216e4b3/dashboard/__init__.py)
