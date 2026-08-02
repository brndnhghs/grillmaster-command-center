# Module: `dashboard`

## Purpose
Process supervisor for the Grillmaster stack. A small FastAPI app (port **7870**) that spawns, monitors, and stops the Image Pipeline (`:7860`) plus the three.js sidecar (`:7862`), and serves a single-page panel showing each service's port, PID, and health. It does NOT render or proxy either service — it launches each as a child process using the repo `.venv` and links out to them.

**Nothing depends on it.** The pipeline server spawns the three.js sidecar itself on first 3D render (`_ensure_threejs_sidecar`), and an already-healthy service is adopted rather than restarted, so both supervisors can run at once. This panel is a convenience for pre-warming and watching services, not a dependency.

## Responsibilities
- Spawn each service as a backgrounded subprocess (repo `.venv` Python — `.venv/Scripts/python.exe` on Windows — with `PYTHONPATH` set to repo root)
- Stop services (kills the whole process tree, not just the listener; `taskkill /T /F` on Windows)
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
Kills the service's process tree via `_kill_tree()` — `os.killpg(os.getpgid(pid), signal.SIGTERM)` on POSIX, `taskkill /T /F` on Windows (which has no process groups) — so any children it spawned die too.

### `_listeners_on_port(port) -> list[int]`
Finds the PIDs holding `tcp:<port>` in LISTEN — `lsof -ti` on POSIX, `netstat -ano` parse on Windows.

### `_reclaim_port(port) -> bool`
`SIGKILL`s every listener from `_listeners_on_port()` (tree-kill). Returns True if something was killed. Prevents "address already in use" on relaunch.

### `_is_healthy(port) -> bool`
Opens `http://127.0.0.1:<port>/health` and requires a 200 — distinguishes "port open but dead" from "actually serving".

### `launch_all() / stop_all()`
Iterate `SERVICES` (pipeline, 3d). `launch_all` spawns everything first, then waits — avoids stacking readiness timeouts.

### `service_status(name, port) -> dict`
Returns process-alive, port-open, and health status for one service.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Dashboard port + per-service status (process, port, health, url) |
| POST | `/api/launch` | Launch all services |
| POST | `/api/stop` | Stop all services |
| POST | `/api/launch/{name}` | Launch one service (`pipeline` / `3d`) |
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
- The repo `.venv` Python (3.12) — has `image_pipeline` on path. On Windows the venv layout is `.venv/Scripts/python.exe`; the module resolves it per-platform (`_IS_WINDOWS`).

## Key Design Notes
- **Forgets `_PROCS` across its own restarts** — so after a dashboard restart, `api_stop_one` also calls `_reclaim_port` to free any orphaned listener.
- **Port reclaim before spawn** — a leftover wedged instance would otherwise block every relaunch.
- **3D sidecar** is a Node.js process (`threejs-sidecar.mjs`), not a Python module; spawned via `_spawn_node`. The pipeline server can also spawn it on demand — `_begin_start` adopts an already-healthy one instead of double-spawning.

## Source
[`dashboard/__init__.py`](https://github.com/brndnhghs/grillmaster-command-center/blob/f689773c452e24fa1bf1bbcf3e6817fb5304c81d/dashboard/__init__.py)
