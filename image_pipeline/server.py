"""FastAPI server for the image-generation pipeline GUI."""
from __future__ import annotations
import asyncio
import base64
import importlib
import io
import json
import queue
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repo root on path for direct invocations
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Auto-register all methods before the registry is queried
from image_pipeline.core import registry
from image_pipeline.core.registry import unregister as _unregister_method, get_id_by_module
from image_pipeline.core.animation import JobCancelled
from image_pipeline.core.graph import GraphExecutor, GraphError
from image_pipeline.core.port_types import all_port_types
import image_pipeline.methods  # noqa: F401

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Stable directory for graph executor caches (persists across runs for dirty-flag skip)
_GRAPH_SESSION_DIR = OUTPUT_ROOT / "graph-session"
_GRAPH_SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Saved named graphs
SAVED_GRAPHS_DIR = OUTPUT_ROOT / "saved-graphs"
SAVED_GRAPHS_DIR.mkdir(exist_ok=True)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

# ── Hot-reload infrastructure ─────────────────────────────────────────

_sse_clients: list[asyncio.Queue] = []
_event_loop: asyncio.AbstractEventLoop | None = None


async def _broadcast_sse(event: str, data: str = "{}"):
    """Push a named SSE event to every connected /api/events client."""
    msg = f"event: {event}\ndata: {data}\n\n"
    for q in list(_sse_clients):
        await q.put(msg)


def _hot_reload_path(filepath: str):
    """Re-import a changed method file and re-register it."""
    path = Path(filepath)
    if path.suffix != '.py':
        return
    try:
        rel = path.relative_to(Path(__file__).parent)
        module_name = "image_pipeline." + str(rel.with_suffix('')).replace('/', '.').replace('\\', '.')
    except ValueError:
        return
    old_id = get_id_by_module(module_name)
    if old_id is not None:
        _unregister_method(old_id)
    if module_name in sys.modules:
        try:
            importlib.reload(sys.modules[module_name])
        except Exception as e:
            print(f"[hot-reload] error reloading {module_name}: {e}")
            return
    else:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            print(f"[hot-reload] error importing {module_name}: {e}")
            return
    print(f"[hot-reload] reloaded {module_name}")
    global _event_loop
    if _event_loop is not None and _event_loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast_sse("node-defs-updated"), _event_loop)


from watchdog.observers import Observer  # noqa: E402
from watchdog.events import FileSystemEventHandler  # noqa: E402


class _MethodWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            _hot_reload_path(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            _hot_reload_path(event.src_path)


_methods_dir = str(Path(__file__).parent / "methods")
_observer = Observer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    _observer.schedule(_MethodWatcher(), _methods_dir, recursive=True)
    _observer.start()
    yield
    _observer.stop()
    _observer.join()


app = FastAPI(title="Image Pipeline", lifespan=lifespan)
app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")

# ── Thread-safe stdout/stderr proxy ──────────────────────────────────
# Each job thread installs its own writer via the proxy instead of replacing
# the global sys.stdout, so concurrent jobs never clobber each other's streams.

class _ThreadDispatchWriter:
    """Proxies writes to a per-thread override; falls back to the real stream."""

    def __init__(self, real):
        self._real = real
        self._local = threading.local()

    def set(self, writer):
        self._local.writer = writer

    def clear(self):
        self._local.writer = None

    def _active(self):
        return getattr(self._local, "writer", None)

    def write(self, s):
        w = self._active()
        (w if w else self._real).write(s)

    def flush(self):
        w = self._active()
        (w if w else self._real).flush()

    def fileno(self):
        return self._real.fileno()


_stdout_proxy = _ThreadDispatchWriter(sys.__stdout__)
_stderr_proxy = _ThreadDispatchWriter(sys.__stderr__)
sys.stdout = _stdout_proxy  # type: ignore[assignment]
sys.stderr = _stderr_proxy  # type: ignore[assignment]


# ── Param-spec enrichment ─────────────────────────────────────────────

def _parse_choices(desc: str, default_val) -> list | None:
    """Infer enumerable choices from a description string.

    Two patterns are detected:
      Paren  : 'label (a/b/c)'  or 'label (a, b, c)' — requires ≥3 items
      Colon  : 'label: a, b, c'                       — requires ≥3 items

    Skips params whose default is a number or bool — those are never string enums.
    """
    if not isinstance(desc, str) or not desc:
        return None
    if isinstance(default_val, (int, float, bool)) or default_val is None:
        return None

    # Paren pattern: (a/b/c) or (a, b, c) with ≥3 items
    m = re.search(r'\(([\w-]+(?:[,/]\s*[\w-]+){2,})\)', desc)
    if m:
        choices = [t.strip() for t in re.split(r'[,/]', m.group(1)) if t.strip()]
        if len(choices) >= 3:
            return choices

    # Colon+comma pattern: 'label: a, b, c' — list must extend to end of description
    m = re.search(r':\s*([\w-]+(?:,\s*[\w-]+){2,})\s*$', desc)
    if m:
        choices = [t.strip() for t in m.group(1).split(',') if t.strip()]
        if len(choices) >= 3:
            return choices

    return None


def _enrich_params(params: dict | None) -> dict | None:
    """Inject 'choices' into param specs where the description encodes an enum list."""
    if not params:
        return params
    result = {}
    for key, spec in params.items():
        if not isinstance(spec, dict) or 'choices' in spec:
            result[key] = spec
            continue
        choices = _parse_choices(spec.get('description', ''), spec.get('default'))
        result[key] = {**spec, 'choices': choices} if choices else spec
    return result


# ── In-memory job store ───────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_JOB_MAX_AGE = 3600  # seconds


def _evict_old_jobs():
    """Remove completed jobs older than _JOB_MAX_AGE to prevent unbounded growth."""
    now = time.time()
    stale = [
        jid for jid, j in _jobs.items()
        if j.get("status") == "done" and now - j["start"] > _JOB_MAX_AGE
    ]
    for jid in stale:
        _jobs.pop(jid, None)


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/")
def serve_ui():
    return FileResponse(str(UI_DIR / "index.html"))


@app.get("/api/methods")
def list_methods():
    all_methods = registry.get_all()
    return [
        {
            "id": meta.id,
            "name": meta.name,
            "category": meta.category,
            "tags": meta.tags,
            "params": _enrich_params(meta.params),
        }
        for meta in sorted(all_methods.values(), key=lambda m: m.id)
    ]


class GenerateRequest(BaseModel):
    method_id: str
    seed: int = 42
    params: dict[str, Any] = {}
    animate: bool = False
    fps: int = 24
    duration: float = 3.0


@app.post("/api/generate")
def generate(req: GenerateRequest):
    _evict_old_jobs()
    job_id = uuid.uuid4().hex[:8]
    out_dir = OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    job: dict = {
        "id": job_id,
        "status": "running",
        "q": queue.Queue(),
        "output_path": None,
        "type": None,
        "start": time.time(),
        "cancel_event": threading.Event(),
    }
    _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, req.method_id, req.seed, req.params or None,
              req.animate, req.fps, req.duration, out_dir),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    if job_id not in _jobs:
        return {"error": "Job not found"}
    _jobs[job_id]["cancel_event"].set()
    return {"status": "cancelling"}


# ── Background worker ─────────────────────────────────────────────────


class _QueueWriter:
    """Redirect stdout/stderr lines into the job queue."""

    def __init__(self, q: queue.Queue, buf: list):
        self._q = q
        self._buf = buf
        self._line = ""

    def write(self, text: str):
        self._line += text
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            msg = line.strip()
            if msg:
                self._buf.append(msg)
                self._q.put(("progress", msg))

    def flush(self):
        if self._line.strip():
            msg = self._line.strip()
            self._line = ""
            self._buf.append(msg)
            self._q.put(("progress", msg))


def _encode_frame(arr) -> str:
    """Encode a numpy array or PIL image as a base64 JPEG string for SSE."""
    from PIL import Image
    import numpy as np
    if isinstance(arr, np.ndarray):
        if arr.dtype != np.uint8:
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = arr
    img = img.convert("RGB")
    # Halve the resolution for a lightweight live preview
    w, h = img.size
    img = img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=65)
    return base64.b64encode(buf.getvalue()).decode()


def _run_job(job_id, method_id, seed, params, animate, fps, duration, out_dir):
    job = _jobs[job_id]
    q = job["q"]
    cancel_event: threading.Event = job["cancel_event"]
    messages: list[str] = []

    meta = registry.get_meta(method_id)
    if meta is None:
        q.put(("error", f"Method '{method_id}' not found"))
        q.put(None)
        return

    writer = _QueueWriter(q, messages)
    _stdout_proxy.set(writer)
    _stderr_proxy.set(writer)

    try:
        if animate:
            # ── Build a frame callback that throttles and encodes to JPEG ──
            last_frame_t = [0.0]

            def _on_frame(arr):
                now = time.time()
                if now - last_frame_t[0] < 0.05:   # cap preview at ~20 fps
                    return
                last_frame_t[0] = now
                q.put(("frame", _encode_frame(arr)))

            # ── Install per-thread job context for capture_frame() ──
            import image_pipeline.core.animation as _anim_mod
            _anim_mod.set_job_context(_on_frame, cancel_event)

            try:
                from image_pipeline.core.animation import animate_method
                out_path = animate_method(
                    meta, out_dir, seed,
                    fps=fps, duration=duration,
                    user_params=params,
                )
            except JobCancelled:
                q.put(("error", "Cancelled"))
                return
            finally:
                _anim_mod.clear_job_context()

            if cancel_event.is_set():
                q.put(("error", "Cancelled"))
                return

            if out_path and Path(out_path).exists():
                job["output_path"] = str(out_path)
                job["type"] = "video"
                q.put(("done", {"output_path": str(out_path), "type": "video"}))
            else:
                q.put(("error", "Animation failed — method has no natural animation frames"))

        else:
            try:
                meta.fn(out_dir, seed, params=params)
            except TypeError as _e:
                if "unexpected keyword argument" not in str(_e):
                    raise
                meta.fn(out_dir, seed)

            if cancel_event.is_set():
                q.put(("error", "Cancelled"))
                return

            pngs = sorted(out_dir.glob("*.png"))
            if pngs:
                out_path = pngs[-1]
                job["output_path"] = str(out_path)
                job["type"] = "image"
                q.put(("done", {"output_path": str(out_path), "type": "image"}))
            else:
                q.put(("error", "No PNG output produced"))

    except Exception as exc:
        q.put(("error", str(exc)))
    finally:
        _stdout_proxy.clear()
        _stderr_proxy.clear()
        job["status"] = "done"
        q.put(None)  # sentinel — tells the SSE generator the stream is over


# ── SSE stream ────────────────────────────────────────────────────────


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: str):
    if job_id not in _jobs:
        def _err():
            yield 'event: error\ndata: {"message": "Job not found"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    job = _jobs[job_id]
    start = job["start"]

    def event_gen():
        while True:
            try:
                item = job["q"].get(timeout=30)
            except queue.Empty:
                # Keep-alive comment so the connection stays open
                yield ": keep-alive\n\n"
                continue

            if item is None:
                break

            event_type, data = item
            elapsed = round(time.time() - start, 1)

            if event_type == "progress":
                payload = json.dumps({"message": data, "elapsed": elapsed})
                yield f"event: progress\ndata: {payload}\n\n"
            elif event_type == "frame":
                # data is already a base64 string — no JSON wrapping needed
                yield f"event: frame\ndata: {data}\n\n"
            elif event_type == "done":
                payload = json.dumps(data)
                yield f"event: done\ndata: {payload}\n\n"
                break
            elif event_type == "error":
                payload = json.dumps({"message": data})
                yield f"event: error\ndata: {payload}\n\n"
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── Hot-reload broadcast endpoint ────────────────────────────────────


@app.get("/api/events")
async def sse_events():
    """SSE stream for server-push events (e.g. node-defs-updated on hot-reload)."""
    async def generator():
        q: asyncio.Queue = asyncio.Queue()
        _sse_clients.append(q)
        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            _sse_clients.remove(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── File result ───────────────────────────────────────────────────────


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in _jobs:
        return {"error": "Job not found"}
    job = _jobs[job_id]
    if not job.get("output_path"):
        return {"error": "No output yet"}
    path = Path(job["output_path"])
    if not path.exists():
        return {"error": "Output file not found"}
    media_type = "video/mp4" if job["type"] == "video" else "image/png"
    return FileResponse(str(path), media_type=media_type,
                        filename=path.name)


# ── Node-graph endpoints ──────────────────────────────────────────────


@app.get("/api/port-types")
def get_port_types():
    return {
        name: {
            "color": spec.color,
            "description": spec.description,
            "accepts_from": spec.accepts_from,
        }
        for name, spec in all_port_types().items()
    }


@app.get("/api/palettes")
def list_palettes():
    from image_pipeline.core.utils import PALETTES
    return list(PALETTES.keys())


@app.get("/api/graph/wire-payload/{job_id}/{src_node_id}")
def get_wire_payload(job_id: str, src_node_id: str):
    """Return the payload manifest for a node output (keys + port types)."""
    if job_id not in _jobs:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "job not found"}, status_code=404)
    node_dir = _GRAPH_SESSION_DIR / src_node_id
    if not node_dir.exists():
        return {"node_id": src_node_id, "payload": {}}
    manifest: dict[str, str] = {}
    pngs = [p for p in node_dir.glob("*.png") if not p.name.startswith("_")]
    if pngs:
        manifest["image"] = "IMAGE"
        manifest["luminance"] = "SCALAR"
    scalars_path = node_dir / "scalars.json"
    if scalars_path.exists():
        try:
            for k in json.loads(scalars_path.read_text()):
                manifest[k] = "SCALAR"
        except Exception:
            pass
    if (node_dir / "field.npy").exists():
        manifest["field"] = "FIELD"
    if (node_dir / "particles.npy").exists():
        manifest["particles"] = "PARTICLES"
    return {"node_id": src_node_id, "payload": manifest}


@app.get("/api/node-defs")
def get_node_defs():
    from image_pipeline.core.graph import get_all_node_defs
    defs = get_all_node_defs()
    for nd in defs.values():
        if nd.get('params'):
            nd['params'] = _enrich_params(nd['params'])
    return defs


# ── Graph save / load endpoints ───────────────────────────────────────


@app.post("/api/graph/save")
async def save_graph(payload: dict):
    name = payload.get("name", "untitled").strip() or "untitled"
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    graph_data = payload.get("graph", {})
    graph_data["name"] = name
    graph_data["saved_at"] = datetime.utcnow().isoformat()
    path = SAVED_GRAPHS_DIR / f"{name}.json"
    path.write_text(json.dumps(graph_data, indent=2))
    return {"ok": True, "name": name}


@app.get("/api/graph/saved")
def list_saved_graphs():
    graphs = []
    for f in sorted(SAVED_GRAPHS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            graphs.append({"name": data.get("name", f.stem), "saved_at": data.get("saved_at", "")})
        except Exception:
            pass
    return graphs


@app.get("/api/graph/saved/{name}")
def load_saved_graph(name: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = SAVED_GRAPHS_DIR / f"{name}.json"
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Graph '{name}' not found")
    return json.loads(path.read_text())


@app.delete("/api/graph/saved/{name}")
def delete_saved_graph(name: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = SAVED_GRAPHS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


class GraphRequest(BaseModel):
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seed: int = 42
    frames: int = 1


@app.post("/api/graph/execute")
def execute_graph(req: GraphRequest):
    _evict_old_jobs()
    job_id = uuid.uuid4().hex[:8]
    out_dir = OUTPUT_ROOT / f"graph-{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    job: dict = {
        "id": job_id,
        "status": "running",
        "q": queue.Queue(),
        "output_path": None,
        "type": "image",
        "start": time.time(),
        "cancel_event": threading.Event(),
    }
    _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_graph_job,
        args=(job_id, req.nodes, req.edges, req.seed, req.frames, out_dir),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


def _run_graph_job(job_id, nodes, edges, seed, frames, out_dir):
    job = _jobs[job_id]
    q = job["q"]
    cancel_event: threading.Event = job["cancel_event"]

    messages: list[str] = []
    writer = _QueueWriter(q, messages)
    _stdout_proxy.set(writer)
    _stderr_proxy.set(writer)

    try:
        from image_pipeline.core.animation import frames_to_mp4
        executor = GraphExecutor(_GRAPH_SESSION_DIR)
        terminal_frames: list = []
        terminal_node_id: str | None = None

        n_frames = max(1, frames)
        all_errors: dict[str, str] = {}
        for frame in range(n_frames):
            if cancel_event.is_set():
                q.put(("error", "Cancelled"))
                return

            print(f"  Frame {frame + 1}/{n_frames}")
            try:
                flat_outputs, terminal_id, frame_errors = executor.execute(
                    nodes, edges, seed, frame=frame, frames=n_frames
                )
            except GraphError as exc:
                q.put(("error", str(exc)))
                return

            for nid, err_text in frame_errors.items():
                all_errors[nid] = err_text
                q.put(("node_error", json.dumps({"nodeId": nid, "error": err_text[:500]})))

            # Honour explicit render flag; fall back to auto-detected terminal
            render_id = next((n["id"] for n in nodes if n.get("render")), None)
            if render_id and render_id in flat_outputs:
                terminal_id = render_id

            if terminal_id is None:
                q.put(("error", "No output node — every node has outgoing connections"))
                return

            terminal_node_id = terminal_id
            arr = (flat_outputs.get(terminal_id) or {}).get("image")
            if arr is not None:
                terminal_frames.append(arr)
                encoded = _encode_frame(arr)
                payload = json.dumps({"frame": frame, "node_id": terminal_id, "data": encoded})
                q.put(("graph_frame", payload))

        # Assemble output file
        if not terminal_frames:
            q.put(("error", "No output produced — terminal node generated no image"))
            return

        if n_frames > 1:
            mp4_path = out_dir / "graph-output.mp4"
            result = frames_to_mp4(iter(terminal_frames), mp4_path, fps=24)
            if result:
                rel_path = result.relative_to(OUTPUT_ROOT).as_posix()
                job["output_path"] = str(result)
                job["type"] = "video"
                q.put(("done", {"rel_path": rel_path, "type": "video", "frames": len(terminal_frames), "errors": all_errors}))
            else:
                q.put(("error", "MP4 assembly failed"))
        else:
            from PIL import Image
            import numpy as np
            arr = terminal_frames[0]
            arr_u8 = (arr.clip(0, 1) * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
            png_path = out_dir / "graph-output.png"
            Image.fromarray(arr_u8).save(str(png_path))
            rel_path = png_path.relative_to(OUTPUT_ROOT).as_posix()
            job["output_path"] = str(png_path)
            job["type"] = "image"
            q.put(("done", {"rel_path": rel_path, "type": "image", "frames": 1, "errors": all_errors}))

    except Exception as exc:
        q.put(("error", str(exc)))
    finally:
        _stdout_proxy.clear()
        _stderr_proxy.clear()
        job["status"] = "done"
        q.put(None)


# Override stream_job to also handle graph_frame events
@app.get("/api/graph/jobs/{job_id}/stream")
def stream_graph_job(job_id: str):
    if job_id not in _jobs:
        def _err():
            yield 'event: error\ndata: {"message": "Job not found"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    job = _jobs[job_id]
    start = job["start"]

    def event_gen():
        while True:
            try:
                item = job["q"].get(timeout=30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue

            if item is None:
                break

            event_type, data = item
            elapsed = round(time.time() - start, 1)

            if event_type == "progress":
                payload = json.dumps({"message": data, "elapsed": elapsed})
                yield f"event: progress\ndata: {payload}\n\n"
            elif event_type == "graph_frame":
                yield f"event: graph_frame\ndata: {data}\n\n"
            elif event_type == "node_error":
                yield f"event: node-error\ndata: {data}\n\n"
            elif event_type == "done":
                payload = json.dumps(data)
                yield f"event: done\ndata: {payload}\n\n"
                break
            elif event_type == "error":
                payload = json.dumps({"message": data})
                yield f"event: error\ndata: {payload}\n\n"
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── NODE DOCTOR endpoints ─────────────────────────────────────────────

_nd_backups: dict[str, tuple[str, str]] = {}  # backup_id → (orig_path, backup_path)

_ND_SYSTEM = """\
You are NODE DOCTOR, a specialist agent embedded in the Grillmaster generative-art node pipeline.

Your job: diagnose, fix, and augment Python method files that power pipeline nodes.

HARD RULES:
- Always output the COMPLETE, runnable Python file — never partial snippets.
- Preserve the method signature exactly: fn(out_dir: Path | str, seed: int, params: dict | None = None)
- Methods must write at least one PNG to out_dir/ if they have an IMAGE output port.
- For sidecar outputs use only: write_scalars(out_dir, {...}), write_field(out_dir, arr), write_particles(out_dir, arr)
  imported from image_pipeline.core.utils.
- You may add new params (must have defaults so existing graphs keep working).
- Never remove existing params or break existing port contracts.

PORT TYPES: IMAGE (H×W×3 float32 [0,1]), SCALAR (float), FIELD (H×W float32), PARTICLES (N×4 float32 [x,y,vx,vy])

RESPONSE FORMAT:
- Reply conversationally first — explain what you changed and why.
- If you are writing new code, end your response with the complete file in a ```python block.
- If you are only answering a question, no code block needed.
"""


def _get_method_path(method_id: str) -> Path | None:
    meta = registry.get_meta(method_id)
    if not meta or not getattr(meta, "module", None):
        return None
    mod = sys.modules.get(meta.module)
    if not mod or not getattr(mod, "__file__", None):
        return None
    return Path(mod.__file__)


@app.get("/api/node-doctor/source/{method_id}")
def nd_get_source(method_id: str):
    path = _get_method_path(method_id)
    if not path or not path.exists():
        return {"source": "", "path": ""}
    return {"source": path.read_text(), "path": str(path)}


@app.post("/api/node-doctor/chat")
async def nd_chat(payload: dict):
    import anthropic as _anthropic
    method_id   = payload.get("method_id", "")
    node_def    = payload.get("node_def", {})
    node_params = payload.get("node_params", {})
    messages    = payload.get("messages", [])

    source = ""
    path = _get_method_path(method_id)
    if path and path.exists():
        source = path.read_text()

    system = _ND_SYSTEM + f"""
--- NODE CONTEXT ---
method_id : {method_id}
name      : {node_def.get('name', '')}
inputs    : {json.dumps(node_def.get('inputs', {}))}
outputs   : {json.dumps(node_def.get('outputs', {}))}
params    : {json.dumps(node_def.get('params', {}))}
current_param_values: {json.dumps(node_params)}

--- CURRENT SOURCE ---
```python
{source}
```
"""

    client = _anthropic.AsyncAnthropic()

    async def generate():
        async with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/node-doctor/apply")
async def nd_apply(payload: dict):
    method_id  = payload.get("method_id", "")
    new_source = payload.get("source", "")
    if not new_source:
        return {"error": "No source provided"}

    path = _get_method_path(method_id)
    if not path or not path.exists():
        return {"error": "Method source file not found"}

    backup_id   = uuid.uuid4().hex[:8]
    backup_path = path.with_suffix(f".nd-bak-{backup_id}.py")
    shutil.copy2(str(path), str(backup_path))
    _nd_backups[backup_id] = (str(path), str(backup_path))

    path.write_text(new_source)
    # watchdog picks up the change and hot-reloads automatically
    return {"ok": True, "backup_id": backup_id}


@app.post("/api/node-doctor/undo/{backup_id}")
async def nd_undo(backup_id: str):
    entry = _nd_backups.pop(backup_id, None)
    if not entry:
        return {"error": "Backup not found"}
    orig_path, backup_path = entry
    shutil.copy2(backup_path, orig_path)
    Path(backup_path).unlink(missing_ok=True)
    return {"ok": True}


# ── Entry point ───────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Image Pipeline server")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--tunnel", action="store_true", help="Open a public ngrok tunnel")
    args = parser.parse_args()

    if args.tunnel:
        from pyngrok import ngrok
        tunnel = ngrok.connect(args.port, bind_tls=True)
        print(f"🌐 Public URL: {tunnel.public_url}")

    uvicorn.run("image_pipeline.server:app", host="0.0.0.0", port=args.port, reload=False, log_config=None)
