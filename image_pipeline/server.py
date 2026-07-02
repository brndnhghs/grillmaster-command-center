"""FastAPI server for the image-generation pipeline GUI."""
from __future__ import annotations
import asyncio
import base64
import importlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
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
from image_pipeline.core.registry import unregister as _unregister_method, get_ids_by_module
from image_pipeline.core.animation import JobCancelled
from image_pipeline.core.graph import GraphExecutor, GraphError
from image_pipeline.core.port_types import all_port_types
from image_pipeline.core import cache as _cache
from image_pipeline.core.quality import check as _quality_check
from image_pipeline.core.postprocess import apply_filter as _apply_filter
from image_pipeline.core.annotator import annotate_image as _annotate_image
import image_pipeline.methods  # noqa: F401

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Stable directory for graph executor caches (persists across runs for dirty-flag skip)
_GRAPH_SESSION_DIR = OUTPUT_ROOT / "graph-session"
_GRAPH_SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Saved named graphs
SAVED_GRAPHS_DIR = OUTPUT_ROOT / "saved-graphs"
SAVED_GRAPHS_DIR.mkdir(exist_ok=True)

# Saved group node presets
SAVED_GROUPS_DIR = OUTPUT_ROOT / "saved-groups"
SAVED_GROUPS_DIR.mkdir(exist_ok=True)

SEQUENCES_DIR = OUTPUT_ROOT / "sequences"
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory keyframe store (per-node keyframe lists, replace semantics)
_keyframe_store: dict[str, list[dict]] = {}

# ── Live preview buffer (MJPEG streaming) ──────────────────────────────
_LIVE_FRAME: bytes | None = None
_LIVE_FRAME_LOCK = threading.Lock()
_LIVE_FRAME_COND = threading.Condition(_LIVE_FRAME_LOCK)  # for MJPEG waiters
_LIVE_FRAME_ID = 0  # monotonic counter so MJPEG can detect new frames


def _push_live_frame(arr):
    """Encode a numpy array as JPEG and store in the live buffer."""
    global _LIVE_FRAME, _LIVE_FRAME_ID
    from PIL import Image
    import numpy as np
    if isinstance(arr, np.ndarray):
        if arr.dtype != np.uint8:
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = arr
    img = img.convert("RGB")
    w, h = img.size
    if w > 1280:
        ratio = 1280 / w
        img = img.resize((1280, int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    with _LIVE_FRAME_LOCK:
        _LIVE_FRAME = buf.getvalue()
        _LIVE_FRAME_ID += 1
        _LIVE_FRAME_COND.notify_all()
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
    for old_id in get_ids_by_module(module_name):
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

# ── Chord Bot sub-application (served at /chordbot/) ─────────────────
from chord_bot.server import app as _chord_app  # noqa: E402
app.mount("/chordbot", _chord_app)

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

    def isatty(self):
        return self._real.isatty()


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

# (seed, frame, edges-digest) of the last single-frame graph run — the
# dirty-flag skip is only valid while these are unchanged; see _run_graph_job.
_last_single_frame_ctx: tuple | None = None

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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/admin/restart")
def admin_restart():
    """Re-exec the server process in place — picks up any source changes."""
    def _exec():
        time.sleep(0.4)  # let the HTTP response drain first
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_exec, daemon=True).start()
    return {"restarting": True}


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
    filter: str | None = None  # postprocess filter spec, e.g. "oil" or '{"effect":"bloom"}'
    demo: bool = False          # overlay param annotations on output image
    width: int = 768
    height: int = 512


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
              req.animate, req.fps, req.duration, out_dir,
              req.filter, req.demo, req.width, req.height),
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


def _run_job(job_id, method_id, seed, params, animate, fps, duration, out_dir, filter_spec=None, demo=False, width=768, height=512):
    from image_pipeline.core.utils import set_canvas as _set_canvas
    _set_canvas(width, height)
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
            # ── Cache check ──
            cached_path = _cache.exists(method_id, seed, out_dir, params)
            if cached_path and not cancel_event.is_set():
                q.put(("progress", "⊛ cache hit"))
                out_path = out_dir / cached_path.name
                shutil.copy2(str(cached_path), str(out_path))
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
                if not pngs:
                    q.put(("error", "No PNG output produced"))
                    return

                out_path = pngs[-1]
                # ── Cache store ──
                _cache.store(method_id, seed, out_path, params)

            # ── Quality check ──
            report = _quality_check(out_path)
            if not report.passed:
                q.put(("progress", f"⚠ quality: {'; '.join(report.issues)}"))

            # ── Post-process filter ──
            if filter_spec:
                try:
                    _apply_filter(out_path, filter_spec)
                except Exception as _fe:
                    q.put(("progress", f"⚠ filter: {_fe}"))

            # ── Demo annotation ──
            if demo:
                try:
                    _annotate_image(method_id, out_path)
                except Exception as _ae:
                    q.put(("progress", f"⚠ demo: {_ae}"))

            job["output_path"] = str(out_path)
            job["type"] = "image"
            q.put(("done", {"output_path": str(out_path), "type": "image"}))

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
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Live preview endpoints (MJPEG stream + polling fallback) ──────────


@app.get("/api/live/stream")
async def live_mjpeg_stream():
    """MJPEG multipart/x-mixed-replace stream of the live preview buffer."""
    def _wait_for_new_frame(last_id):
        """Block until a new frame is available or timeout."""
        with _LIVE_FRAME_COND:
            if _LIVE_FRAME_ID == last_id:
                _LIVE_FRAME_COND.wait(timeout=1.0)

    async def generate():
        last_id = -1
        loop = asyncio.get_event_loop()
        while True:
            if _LIVE_FRAME_ID == last_id:
                await loop.run_in_executor(None, _wait_for_new_frame, last_id)
            with _LIVE_FRAME_LOCK:
                fid = _LIVE_FRAME_ID
                data = _LIVE_FRAME
            if data is not None and fid != last_id:
                last_id = fid
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n"
                    b"\r\n" + data + b"\r\n"
                )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/live/frame.jpg")
def live_frame_jpg():
    """Polling fallback — returns the latest live frame as a JPEG.
    The browser polls this with ?t=timestamp to bypass cache.
    """
    with _LIVE_FRAME_LOCK:
        data = _LIVE_FRAME
    if data is None:
        from fastapi.responses import Response
        return Response(status_code=204)
    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
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


# ── Group node preset endpoints ───────────────────────────────────────


@app.post("/api/groups/save")
async def save_group(payload: dict):
    name = payload.get("name", "untitled").strip() or "untitled"
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    group_data = {
        "name": name,
        "subgraph": payload.get("subgraph", {}),
        "exposed_inputs": payload.get("exposed_inputs", []),
        "exposed_outputs": payload.get("exposed_outputs", []),
        "saved_at": datetime.utcnow().isoformat(),
    }
    path = SAVED_GROUPS_DIR / f"{name}.json"
    path.write_text(json.dumps(group_data, indent=2))
    return {"ok": True, "name": name}


@app.get("/api/groups")
def list_groups():
    groups = []
    for f in sorted(SAVED_GROUPS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            groups.append({"name": data.get("name", f.stem), "saved_at": data.get("saved_at", "")})
        except Exception:
            pass
    return groups


@app.get("/api/groups/{name}")
def get_group(name: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = SAVED_GROUPS_DIR / f"{name}.json"
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Group '{name}' not found")
    return json.loads(path.read_text())


@app.delete("/api/groups/{name}")
def delete_group(name: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = SAVED_GROUPS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


class GraphRequest(BaseModel):
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seed: int = 42
    frames: int = 1
    frame: int = 0
    width: int = 768
    height: int = 512


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
        args=(job_id, req.nodes, req.edges, req.seed, req.frames, req.frame, out_dir,
              req.width, req.height),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


# ── Live sim: continuous graph execution ──────────────────────────
_live_sim_lock = threading.Lock()
_live_sim_cancel = threading.Event()
_live_sim_thread: threading.Thread | None = None


@app.post("/api/graph/live")
def live_graph_sim(req: GraphRequest):
    """Start or stop a continuous graph simulation.

    frames == 0 stops; anything else (re)starts. Only one live loop runs at
    a time — starting while one is active stops the old loop first, so
    re-POSTing with an edited graph hot-swaps it.
    """
    global _live_sim_cancel, _live_sim_thread
    with _live_sim_lock:
        # Stop any existing loop (both for stop requests and restarts)
        if _live_sim_thread is not None and _live_sim_thread.is_alive():
            _live_sim_cancel.set()
            _live_sim_thread.join(timeout=5.0)
        _live_sim_thread = None
        if req.frames == 0:
            return {"status": "stopped"}

        cancel = threading.Event()
        nodes, edges, seed = req.nodes, req.edges, req.seed
        width, height = req.width, req.height

        def _live_loop():
            from image_pipeline.core.graph import GraphExecutor
            from image_pipeline.core.utils import set_canvas as _set_canvas
            _set_canvas(width, height)
            executor = GraphExecutor(OUTPUT_ROOT / "_live_sim", in_memory=True)
            frame = 0
            consecutive_errors = 0
            while not cancel.is_set():
                try:
                    flat_outputs, terminal_id, node_errors = executor.execute(
                        nodes, edges, seed, frame=frame, frames=1
                    )
                    for nid, err in node_errors.items():
                        print(f"[live-sim] node {nid} error:\n{err}")
                    render_id = next((n["id"] for n in nodes if n.get("render")), None)
                    if render_id and render_id in flat_outputs:
                        terminal_id = render_id
                    arr = (flat_outputs.get(terminal_id) or {}).get("image") if terminal_id else None
                    if arr is not None:
                        _push_live_frame(arr)
                    frame += 1
                    consecutive_errors = 0
                except Exception:
                    import traceback as _tb
                    print(f"[live-sim] frame {frame} failed:\n{_tb.format_exc(limit=6)}")
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("[live-sim] 10 consecutive failures — stopping loop")
                        break
                    time.sleep(0.5)  # don't spin on a persistently broken graph
                time.sleep(0.01)

        _live_sim_cancel = cancel
        _live_sim_thread = threading.Thread(target=_live_loop, daemon=True)
        _live_sim_thread.start()
        return {"status": "running"}


@app.get("/api/graph/live/status")
def live_graph_status():
    return {"running": _live_sim_thread is not None and _live_sim_thread.is_alive()}


def _run_graph_job(job_id, nodes, edges, seed, frames, start_frame, out_dir, width=768, height=512):
    from image_pipeline.core.utils import set_canvas as _set_canvas
    _set_canvas(width, height)
    job = _jobs[job_id]
    q = job["q"]
    cancel_event: threading.Event = job["cancel_event"]

    messages: list[str] = []
    writer = _QueueWriter(q, messages)
    _stdout_proxy.set(writer)
    _stderr_proxy.set(writer)

    try:
        from image_pipeline.core.animation import frames_to_mp4
        executor = GraphExecutor(_GRAPH_SESSION_DIR, in_memory=True)
        terminal_frames: list = []
        terminal_node_id: str | None = None

        n_frames = max(1, frames)
        all_errors: dict[str, str] = {}

        # ── Dirty-flag handling ──────────────────────────────────────────
        # Multi-frame renders force everything dirty: the executor's disk
        # cache (_GRAPH_SESSION_DIR) persists across jobs, and reusing the
        # previous run's PNG for every frame would produce a static image
        # instead of re-cooking the animation.
        # Single-frame runs honor the client's dirty flags (selective
        # recooking) — unless the seed, frame, or wiring changed since the
        # last run, which invalidates every cached output regardless of
        # param edits (the client only dirties nodes on param change).
        global _last_single_frame_ctx
        _run_ctx = (seed, start_frame, json.dumps(edges, sort_keys=True, default=str))
        if n_frames > 1 or _last_single_frame_ctx != _run_ctx:
            for n in nodes:
                n["dirty"] = True
        _last_single_frame_ctx = _run_ctx if n_frames == 1 else None

        # ── Determine timeline sequence name ──────────────────────────
        # Use the timeline name from the graph, or auto-generate one.
        tl_node = next((n for n in nodes if n.get("method_id") == "__timeline__"), None)
        seq_name = (tl_node.get("params") or {}).get("name", "") if tl_node else ""
        if not seq_name:
            seq_name = f"run-{job_id}"
        seq_dir = SEQUENCES_DIR / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        for frame in range(start_frame, start_frame + n_frames):
            if cancel_event.is_set():
                q.put(("error", "Cancelled"))
                return

            print(f"  Frame {frame - start_frame + 1}/{n_frames}")
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

                # ── Save individual frame to sequence directory ──
                from PIL import Image as _PILS
                import numpy as np
                png_path = seq_dir / f"frame_{frame:04d}.png"
                _PILS.fromarray(
                    (arr.clip(0, 1) * 255).astype(np.uint8)
                ).save(str(png_path))

        # Assemble output file
        if not terminal_frames:
            _term_desc = f"'{terminal_node_id}'" if terminal_node_id else "(none found)"
            q.put(("error",
                   f"No output produced — terminal node {_term_desc} generated no image. "
                   "It may be a data-only node (Timeline, LFO, Math, …); "
                   "set the render flag on an image-producing node."))
            return

        if n_frames > 1:
            mp4_path = seq_dir / "output.mp4"
            result = frames_to_mp4(iter(terminal_frames), mp4_path, fps=24)
            if result:
                rel_path = result.relative_to(OUTPUT_ROOT).as_posix()
                job["output_path"] = str(result)
                job["type"] = "video"
                q.put(("done", {"rel_path": rel_path, "type": "video", "frames": len(terminal_frames), "errors": all_errors, "seq_name": seq_name}))
            else:
                q.put(("error", "MP4 assembly failed"))
        else:
            from PIL import Image
            import numpy as np
            arr = terminal_frames[0]
            arr_u8 = (arr.clip(0, 1) * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
            png_path = seq_dir / "frame_0000.png"
            Image.fromarray(arr_u8).save(str(png_path))
            rel_path = png_path.relative_to(OUTPUT_ROOT).as_posix()
            job["output_path"] = str(png_path)
            job["type"] = "image"
            q.put(("done", {"rel_path": rel_path, "type": "image", "frames": 1, "errors": all_errors, "seq_name": seq_name}))

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


# ── Sequence render endpoints ─────────────────────────────────────────


def _interpolate_params(node_params: dict, anim_params: dict, frame: int, start_frame: int, end_frame: int) -> dict:
    """Linearly interpolate animated params across the frame range."""
    if not anim_params or end_frame <= start_frame:
        return node_params
    t = (frame - start_frame) / (end_frame - start_frame)
    result = dict(node_params)
    for key, anim in anim_params.items():
        if anim.get("enabled"):
            result[key] = anim["from"] + (anim["to"] - anim["from"]) * t
    return result


class SequenceRequest(BaseModel):
    graph: dict[str, Any]
    start_frame: int = 0
    end_frame: int = 47
    fps: int = 24
    output_name: str = "sequence"
    width: int = 768
    height: int = 512


@app.post("/api/graph/render-sequence")
async def render_sequence(req: SequenceRequest):
    """SSE stream that renders a frame range and saves PNGs to sequences/<name>/."""
    output_name = re.sub(r'[^a-zA-Z0-9_-]', '_', req.output_name) or "sequence"
    seq_dir = SEQUENCES_DIR / output_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    work_dir = seq_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    nodes = req.graph.get("nodes", [])
    edges = req.graph.get("edges", [])
    seed = req.graph.get("seed", 42)
    start_frame = req.start_frame
    end_frame = req.end_frame
    total_frames = end_frame - start_frame + 1

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _seq_worker():
        from image_pipeline.core.utils import set_canvas as _set_canvas
        _set_canvas(req.width, req.height)
        executor = GraphExecutor(work_dir, fps=req.fps, in_memory=True)
        n_frames = end_frame - start_frame + 1

        # Extract per-node animParams (stripped from GraphNode schema)
        node_anim_params = {n["id"]: n.get("animParams", {}) for n in nodes}

        for frame in range(start_frame, end_frame + 1):
            # Build frame-specific node list with interpolated params
            frame_nodes = []
            for n in nodes:
                nid = n["id"]
                anim = node_anim_params.get(nid) or {}
                has_active_anim = any(v.get("enabled") for v in anim.values())
                if has_active_anim:
                    frame_params = _interpolate_params(n.get("params", {}), anim, frame, start_frame, end_frame)
                    frame_nodes.append({**n, "params": frame_params, "dirty": True})
                else:
                    frame_nodes.append(n)

            try:
                flat_outputs, terminal_id, frame_errors = executor.execute(
                    frame_nodes, edges, seed, frame=frame, frames=n_frames
                )
                render_id = next((n["id"] for n in nodes if n.get("render")), None)
                if render_id and render_id in flat_outputs:
                    terminal_id = render_id

                arr = (flat_outputs.get(terminal_id) or {}).get("image") if terminal_id else None
                frame_path = ""
                if arr is not None:
                    from PIL import Image as _PILS
                    import numpy as np
                    png_path = seq_dir / f"frame_{frame:04d}.png"
                    _PILS.fromarray(
                        (arr.clip(0, 1) * 255).astype(np.uint8)
                    ).save(str(png_path))
                    frame_path = str(png_path)

                payload = json.dumps({"frame": frame, "total": total_frames, "path": frame_path})
                asyncio.run_coroutine_threadsafe(
                    event_queue.put(f"event: frame-done\ndata: {payload}\n\n"), loop
                )
            except Exception as exc:
                err_payload = json.dumps({"frame": frame, "error": str(exc)})
                asyncio.run_coroutine_threadsafe(
                    event_queue.put(f"event: frame-error\ndata: {err_payload}\n\n"), loop
                )

        done_payload = json.dumps({
            "name": output_name,
            "frames": total_frames,
            "dir": str(seq_dir),
        })
        asyncio.run_coroutine_threadsafe(
            event_queue.put(f"event: sequence-done\ndata: {done_payload}\n\n"), loop
        )
        asyncio.run_coroutine_threadsafe(event_queue.put(None), loop)

    threading.Thread(target=_seq_worker, daemon=True).start()

    async def _seq_generator():
        while True:
            msg = await event_queue.get()
            if msg is None:
                break
            yield msg

    return StreamingResponse(
        _seq_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sequences")
def list_sequences():
    result = []
    for seq_dir in sorted(SEQUENCES_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not seq_dir.is_dir() or seq_dir.name.startswith("_"):
            continue
        frames = sorted(seq_dir.glob("frame_*.png"))
        result.append({
            "name": seq_dir.name,
            "frame_count": len(frames),
            "created_at": datetime.fromtimestamp(seq_dir.stat().st_mtime).isoformat(),
        })
    return result


@app.get("/api/sequences/{name}/{frame}")
def get_sequence_frame(name: str, frame: int):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    png_path = SEQUENCES_DIR / name / f"frame_{frame:04d}.png"
    if not png_path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Frame {frame} not found in sequence '{name}'")
    # Serve as JPEG for faster transfer — PNGs are 5-10x larger
    jpg_path = png_path.with_suffix(".jpg")
    if not jpg_path.exists():
        from PIL import Image
        img = Image.open(str(png_path)).convert("RGB")
        img.save(str(jpg_path), format="JPEG", quality=85)
    return FileResponse(
        str(jpg_path),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@app.delete("/api/sequences/{name}")
def delete_sequence(name: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    seq_dir = SEQUENCES_DIR / name
    if seq_dir.exists():
        shutil.rmtree(str(seq_dir))
    return {"ok": True}


class EncodeRequest(BaseModel):
    fps: int = 24
    format: str = "mp4"


@app.post("/api/sequences/{name}/encode")
def encode_sequence(name: str, req: EncodeRequest):
    """Encode a rendered sequence of PNGs to mp4 or gif using ffmpeg."""
    if not shutil.which("ffmpeg"):
        return {"ok": False, "error": "ffmpeg not found"}

    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    seq_dir = SEQUENCES_DIR / name
    if not seq_dir.exists():
        return {"ok": False, "error": f"Sequence '{name}' not found"}

    frames = sorted(seq_dir.glob("frame_*.png"), key=lambda p: p.name)
    if not frames:
        return {"ok": False, "error": "No frames found in sequence"}

    fmt = req.format.lower()
    if fmt not in ("mp4", "gif"):
        return {"ok": False, "error": f"Unsupported format '{fmt}' — use mp4 or gif"}

    out_file = seq_dir / f"output.{fmt}"
    input_pattern = str(seq_dir / "frame_%04d.png")

    if fmt == "mp4":
        cmd = [
            "ffmpeg", "-framerate", str(req.fps),
            "-i", input_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-y", str(out_file),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.decode(errors="replace")[:500]}
    else:
        # GIF: 2-pass (palette + encode)
        palette_path = seq_dir / "_palette.png"
        cmd1 = [
            "ffmpeg", "-framerate", str(req.fps),
            "-i", input_pattern,
            "-vf", "palettegen",
            "-y", str(palette_path),
        ]
        r1 = subprocess.run(cmd1, capture_output=True)
        if r1.returncode != 0:
            return {"ok": False, "error": r1.stderr.decode(errors="replace")[:500]}
        cmd2 = [
            "ffmpeg", "-framerate", str(req.fps),
            "-i", input_pattern,
            "-i", str(palette_path),
            "-filter_complex", "[0:v][1:v]paletteuse",
            "-y", str(out_file),
        ]
        result = subprocess.run(cmd2, capture_output=True)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.decode(errors="replace")[:500]}

    return {"ok": True, "path": f"/api/sequences/{name}/video.{fmt}"}


@app.get("/api/sequences/{name}/video.{ext}")
def get_sequence_video(name: str, ext: str):
    """Serve an encoded video file for a sequence."""
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    if ext not in ("mp4", "gif"):
        from fastapi import HTTPException
        raise HTTPException(400, "Unsupported format — use mp4 or gif")
    video_path = SEQUENCES_DIR / name / f"output.{ext}"
    if not video_path.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Video not found for sequence '{name}' — encode first")
    media_type = "video/mp4" if ext == "mp4" else "image/gif"
    return FileResponse(str(video_path), media_type=media_type, filename=f"{name}.{ext}")


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


_ND_RUNNER   = Path(__file__).resolve().parent / "nd_runner.py"
_HERMES_PY   = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"


@app.post("/api/node-doctor/chat")
async def nd_chat(payload: dict):
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

    stdin_bytes = json.dumps({"system_prompt": system, "messages": messages}).encode()

    async def generate():
        try:
            proc = await asyncio.create_subprocess_exec(
                str(_HERMES_PY), str(_ND_RUNNER),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes), timeout=120
            )
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'text': '⚠ NODE DOCTOR timed out'})}\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'text': f'⚠ subprocess error: {exc}'})}\n\n"
            return

        if proc.returncode != 0 and not stdout.strip():
            err = stderr.decode()[:500] if stderr else "no output"
            yield f"data: {json.dumps({'text': f'⚠ runner failed: {err}'})}\n\n"
            return

        for raw in stdout.decode().splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "text" in d:
                    yield f"data: {json.dumps({'text': d['text']})}\n\n"
                elif "error" in d:
                    yield f"data: {json.dumps({'text': '⚠ ' + d['error']})}\n\n"
                elif d.get("done"):
                    yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception:
                pass

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


# ── Keyframe API endpoints ────────────────────────────────────────────


class KeyframeRequest(BaseModel):
    node_id: str
    keyframes: list[dict] = []


@app.post("/api/graph/keyframes")
def set_keyframes(req: KeyframeRequest):
    """Store keyframes for a node. The frontend sends the full keyframe list
    for a node on every edit (replace semantics)."""
    # Keyframes are stored per-node in a simple in-memory dict.
    # In a full implementation these would be persisted with the graph.
    _keyframe_store[req.node_id] = req.keyframes
    return {"ok": True, "count": len(req.keyframes)}


@app.get("/api/graph/keyframes/{node_id}")
def get_keyframes(node_id: str):
    """Retrieve keyframes for a node."""
    kfs = _keyframe_store.get(node_id, [])
    return {"node_id": node_id, "keyframes": kfs}


@app.get("/api/easing-presets")
def get_easing_presets():
    """Return available easing presets for the UI."""
    from image_pipeline.core.easing import EASING_PRESETS
    return {"presets": [{"id": p[0], "name": p[1], "description": p[2]} for p in EASING_PRESETS]}


# ── Node Tester API endpoints ──────────────────────────────────────────


_TEST_OUT_DIR = OUTPUT_ROOT / "node-tests"
_TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
_test_in_progress = False
_test_cancelled = False


@app.post("/api/node-tester/run")
async def nt_run(payload: dict):
    """Run tests on all (or specified) methods. Returns SSE stream of progress + final report."""
    global _test_in_progress, _test_cancelled
    if _test_in_progress:
        return {"error": "Test run already in progress"}

    method_ids = payload.get("method_ids")  # None = all
    include_edge = payload.get("include_edge_cases", True)

    from image_pipeline.core.node_tester import run_all_tests

    async def generate():
        global _test_in_progress, _test_cancelled
        _test_in_progress = True
        _test_cancelled = False

        def progress(mid, name, status, param_set):
            if _test_cancelled:
                raise RuntimeError("Cancelled")
            # Schedule SSE push onto the event loop
            msg = json.dumps({"method_id": mid, "method_name": name, "status": status, "param_set": param_set})
            asyncio.run_coroutine_threadsafe(
                _broadcast_sse("test-progress", msg), _event_loop
            )

        try:
            report = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_all_tests(
                    _TEST_OUT_DIR,
                    method_ids=method_ids,
                    include_edge_cases=include_edge,
                    progress_callback=progress,
                ),
            )
            # Save report to disk
            (_TEST_OUT_DIR / "last_report.json").write_text(json.dumps(report.to_dict()))
            yield f"data: {json.dumps({'done': True, 'report': report.to_dict()})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'done': True, 'error': str(exc)[:500]})}\n\n"
        finally:
            _test_in_progress = False
            _test_cancelled = False

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/node-tester/cancel")
def nt_cancel():
    """Cancel a running test suite."""
    global _test_cancelled
    _test_cancelled = True
    return {"ok": True}


@app.get("/api/node-tester/status")
def nt_status():
    """Check if a test run is in progress."""
    return {"in_progress": _test_in_progress}


@app.get("/api/node-tester/report")
def nt_get_report():
    """Return the most recent test report."""
    report_path = _TEST_OUT_DIR / "last_report.json"
    if not report_path.exists():
        return {"report": None}
    return {"report": json.loads(report_path.read_text())}


@app.post("/api/node-tester/batch-apply")
async def nt_batch_apply(payload: dict):
    """Apply Node Doctor fixes to multiple failing methods at once.

    Payload: {fixes: [{method_id, source, backup_id?}]}
    Returns: {applied: int, failed: [{method_id, error}]}
    """
    fixes = payload.get("fixes", [])
    from image_pipeline.core.node_tester import batch_apply_fixes
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: batch_apply_fixes(fixes, _TEST_OUT_DIR)
    )
    # Broadcast hot-reload for each applied fix
    if result["applied"] > 0:
        await _broadcast_sse("node-defs-updated")
    return result


# ── Test Node report endpoint ────────────────────────────────────────────


@app.get("/api/test-node/report/{node_id}")
def tn_get_report(node_id: str):
    """Read the test_report.json from a Test Node's output directory."""
    report_path = _GRAPH_SESSION_DIR / node_id / "test_report.json"
    if not report_path.exists():
        return {"report": None}
    return {"report": json.loads(report_path.read_text())}


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
