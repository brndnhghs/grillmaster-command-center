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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root on path for direct invocations
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Auto-register all methods before the registry is queried
from image_pipeline.core import registry
from image_pipeline.core.registry import unregister as _unregister_method, get_ids_by_module
from image_pipeline.core.animation import JobCancelled
from image_pipeline.core.graph import GraphExecutor, GraphError, _THREEJS_3D_NODE_DEFS, clear_node_defs_cache
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

# ── Shared graph document store (single source of truth) ───────────
# Both the browser editor and agents read/write the SAME graph doc here.
# The live-sim loop keys off this doc, so clearing it (user or agent) stops
# the render. Persisted to disk so it survives restarts.
GRAPHS_DIR = OUTPUT_ROOT / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

_graph_store_lock = threading.Lock()
_graph_docs: dict[str, dict] = {}


def _graph_path(gid: str) -> Path:
    gid = re.sub(r'[^a-zA-Z0-9_-]', '_', gid)
    return GRAPHS_DIR / f"{gid}.json"


def _graph_default() -> dict:
    return {
        "id": "active",
        "nodes": [],
        "edges": [],
        "canvas": {"w": 768, "h": 512},
        "meta": {"updated_at": None, "updated_by": None},
    }


def _load_graph_doc(gid: str) -> dict:
    """Return the graph doc, loading from disk (or creating a default) once."""
    if gid in _graph_docs:
        return _graph_docs[gid]
    p = _graph_path(gid)
    doc = None
    if p.exists():
        try:
            doc = json.loads(p.read_text())
        except Exception:
            doc = None
    if doc is None:
        doc = _graph_default()
        doc["id"] = gid
    # Normalize shape
    doc.setdefault("nodes", [])
    doc.setdefault("edges", [])
    doc.setdefault("canvas", {"w": 768, "h": 512})
    doc.setdefault("meta", {})
    _graph_docs[gid] = doc
    return doc


def _persist_graph_doc(doc: dict) -> None:
    gid = doc.get("id", "active")
    _graph_docs[gid] = doc
    try:
        _graph_path(gid).write_text(json.dumps(doc, indent=2))
    except Exception as exc:
        print(f"[graph-store] write error for '{gid}': {exc}")


def _touch_graph_meta(doc: dict, by: str | None) -> None:
    doc["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    if by is not None:
        doc["meta"]["updated_by"] = by

# ── Live preview buffer (MJPEG streaming) ──────────────────────────────
_LIVE_FRAME: bytes | None = None
_LIVE_FRAME_LOCK = threading.Lock()
_LIVE_FRAME_COND = threading.Condition(_LIVE_FRAME_LOCK)  # for MJPEG waiters
_LIVE_FRAME_ID = 0  # monotonic counter so MJPEG can detect new frames

# ── WebSocket live broadcast ────────────────────────────────────────────
_WS_CLIENTS: set[WebSocket] = set()
_WS_LOCK = threading.Lock()

# ── Graph-mutation broadcast (user <-> agent shared doc) ────────────────
# Separate WS channel so graph edits propagate to all viewers without
# touching the frame stream.
_GRAPH_WS_CLIENTS: set[WebSocket] = set()
_GRAPH_WS_LOCK = threading.Lock()


def _broadcast_graph_event(event: str, data: dict) -> None:
    """Notify all graph-WS subscribers of a mutation (user or agent origin)."""
    payload = json.dumps({"event": event, **data}, separators=(",", ":"))
    with _GRAPH_WS_LOCK:
        dead = []
        for ws in list(_GRAPH_WS_CLIENTS):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(payload), _event_loop)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _GRAPH_WS_CLIENTS.discard(ws)
    # Also surface over SSE for non-WS clients
    try:
        asyncio.run_coroutine_threadsafe(
            _broadcast_sse("graph", payload), _event_loop
        )
    except Exception:
        pass


def _broadcast_ws_frame(jpeg_bytes: bytes, ws_meta: dict | None = None):
    """Send a per-frame JSON message to all connected WebSocket clients.

    The message shape:
      {"frame": int, "cook_ms": float, "fps": float,
       "node_timings": {id→ms}, "node_names": {id→name},
       "node_errors": {id→str},
       "canvas_w": int, "canvas_h": int,
       "img": "<base64-jpeg>"}

    Falls back to binary JPEG if ws_meta is None (should not happen in normal use).
    """
    if not _WS_CLIENTS:
        return
    if ws_meta is None:
        # Legacy binary fallback (no metadata)
        payload_text = None
        payload_bytes = jpeg_bytes
    else:
        msg = {
            "frame":        ws_meta.get("frame", 0),
            "cook_ms":      ws_meta.get("cook_ms", 0.0),
            "fps":          ws_meta.get("fps", 0.0),
            "node_timings": ws_meta.get("node_timings", {}),
            "node_names":   ws_meta.get("node_names", {}),
            "node_errors":  ws_meta.get("node_errors", {}),
            "gpu_nodes":    ws_meta.get("gpu_nodes", 0),
            "cpu_nodes":    ws_meta.get("cpu_nodes", 0),
            "mem_edges":    ws_meta.get("mem_edges", 0),
            "disk_edges":   ws_meta.get("disk_edges", 0),
            "edge_transport": ws_meta.get("edge_transport", {}),
            "canvas_w":     ws_meta.get("canvas_w", 0),
            "canvas_h":     ws_meta.get("canvas_h", 0),
            "img":          base64.b64encode(jpeg_bytes).decode("ascii"),
        }
        payload_text  = json.dumps(msg, separators=(",", ":"))
        payload_bytes = None

    dead: list[WebSocket] = []
    with _WS_LOCK:
        for ws in list(_WS_CLIENTS):
            try:
                if payload_text is not None:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(payload_text), _event_loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_bytes(payload_bytes), _event_loop
                    )
            except Exception:
                dead.append(ws)
        for ws in dead:
            _WS_CLIENTS.discard(ws)


def _encode_jpeg(arr, *, quality: int, max_width: int = 0, halve: bool = False) -> bytes:
    """Encode an image (float [0,1] / uint8 ndarray, or PIL) to JPEG bytes.

    Uses cv2 (libjpeg-turbo; faster encode and a much cheaper INTER_AREA
    downscale than PIL's LANCZOS) and falls back to PIL if cv2 is missing.
    """
    import numpy as np
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr.convert("RGB"))
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    try:
        import cv2
        if halve:
            arr = cv2.resize(arr, (max(1, w // 2), max(1, h // 2)),
                             interpolation=cv2.INTER_AREA)
        elif max_width and w > max_width:
            arr = cv2.resize(arr, (max_width, int(h * max_width / w)),
                             interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(
            ".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok:
            return enc.tobytes()
    except Exception:
        pass
    # PIL fallback
    from PIL import Image
    img = Image.fromarray(arr)
    if halve:
        img = img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    elif max_width and w > max_width:
        img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _push_live_frame(arr, ws_meta: dict | None = None):
    """Encode a numpy array as JPEG, update the MJPEG buffer, broadcast to WS.

    ws_meta: diagnostics snapshot to embed in the WS message.  When provided
    the WS receives a JSON frame (image + metadata).  MJPEG clients always
    receive raw JPEG regardless.
    """
    global _LIVE_FRAME, _LIVE_FRAME_ID
    jpeg_bytes = _encode_jpeg(arr, quality=85, max_width=1280)
    with _LIVE_FRAME_LOCK:
        _LIVE_FRAME = jpeg_bytes
        _LIVE_FRAME_ID += 1
        _LIVE_FRAME_COND.notify_all()
    _broadcast_ws_frame(jpeg_bytes, ws_meta)
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
    clear_node_defs_cache()
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


# ── Mutating-endpoint auth ────────────────────────────────────────────
# Endpoints that write method source, restart the process, or spawn cook
# loops are unauthenticated by default (localhost use). When the server is
# exposed (e.g. --tunnel), set GRILLMASTER_API_TOKEN; those endpoints then
# require the X-Api-Token header. The UI attaches it automatically from
# localStorage['api-token'].
API_TOKEN = os.environ.get("GRILLMASTER_API_TOKEN", "")


def require_token(request: Request):
    if API_TOKEN and request.headers.get("x-api-token") != API_TOKEN:
        raise HTTPException(401, "Missing or invalid X-Api-Token header")


app = FastAPI(title="Image Pipeline", lifespan=lifespan)
app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")
# Static UI assets (vendored three.js, client-side executor modules). Additive —
# purely for serving front-end files; does not touch the render/export pipeline.
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
# User-uploaded model/texture assets (USD, GLTF, images) — see /api/assets/upload.
ASSETS_DIR = OUTPUT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# ── Chord Bot sub-application (served at /chordbot/) ─────────────────
# Guarded import: chord_bot is an independent sibling app. A failure there
# (missing deps, import error) must not take down the image server boot path.
try:
    from chord_bot.server import app as _chord_app  # noqa: E402
    app.mount("/chordbot", _chord_app)
except Exception as _chord_err:  # noqa: BLE001 — keep the image editor alive
    print(f"[warn] chord_bot not mounted at /chordbot: {_chord_err}")

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


@app.get("/shootout")
def serve_shootout_ui():
    return FileResponse(str(UI_DIR / "shootout.html"))


@app.get("/tune")
def serve_tune_ui():
    return FileResponse(str(UI_DIR / "tune.html"))


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/admin/restart", dependencies=[Depends(require_token)])
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
    """Encode a numpy array or PIL image as a base64 JPEG string for SSE.
    Half resolution for a lightweight preview; cv2-backed encode."""
    return base64.b64encode(_encode_jpeg(arr, quality=65, halve=True)).decode()


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


@app.websocket("/api/live/ws")
async def live_websocket(websocket: WebSocket):
    """WebSocket endpoint for smooth live preview.

    Sends JSON frames: {frame, cook_ms, fps, node_timings, node_names,
    node_errors, canvas_w, canvas_h, img} where img is base64-encoded JPEG.
    The live loop delivers frames via _broadcast_ws_frame; this handler just
    keeps the connection open and accepts keepalive pings from the client.
    """
    await websocket.accept()
    with _WS_LOCK:
        _WS_CLIENTS.add(websocket)
    try:
        # Keep the connection open — JSON frames arrive via _broadcast_ws_frame
        while True:
            await websocket.receive_text()  # keepalive / ping
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _WS_LOCK:
            _WS_CLIENTS.discard(websocket)


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
        manifest["luminance"] = "FIELD"  # per-pixel (H,W) grayscale
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


@app.get("/api/shader-sources")
def get_shader_sources():
    """Read-only: WebGL2 fragment sources for every GPU shader, from the SAME
    GLSL body the server compiles (shader parity layer). Lets the browser
    executor render GPU shader nodes client-side. Additive — does not touch the
    server render/export path."""
    from image_pipeline.core.shaders import shader_sources_for_client
    from image_pipeline.methods.gpu_shaders import GPU_SHADER_NODE_MAP
    bundle = shader_sources_for_client()
    bundle["node_map"] = GPU_SHADER_NODE_MAP  # method_id -> {shader, type}
    return bundle


# ── Graph save / load endpoints ───────────────────────────────────────

# ── Shared graph document API (single source of truth for user + agent) ──
# Endpoints consume/produce the SAME node/edge shape that /api/graph/execute
# and /api/graph/live already accept, so no new graph format is introduced.

class GraphPatch(BaseModel):
    """Fine-grained mutation ops from the agent (or UI)."""
    ops: list[dict] = []          # list of {op, ...} ops
    by: str | None = None         # "user" | "agent" | arbitrary tag


# NOTE: every static /api/graph/<literal> route MUST be registered before the
# dynamic /api/graph/{gid} routes below — FastAPI matches in registration
# order, so a later "/api/graph/saved" would be captured as gid="saved".
# (That exact shadowing silently broke the saved-graphs list and the
# diagnostics endpoint when the shared-doc store landed.)

@app.post("/api/graph/save")
async def save_graph(payload: dict):
    """Save a named graph snapshot (the UI's Save button)."""
    name = (payload.get("name", "untitled") or "untitled").strip() or "untitled"
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    graph_data = payload.get("graph", {})
    graph_data["name"] = name
    graph_data["saved_at"] = datetime.now(timezone.utc).isoformat()
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
        raise HTTPException(404, f"Graph '{name}' not found")
    return json.loads(path.read_text())


@app.delete("/api/graph/saved/{name}")
def delete_saved_graph(name: str):
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = SAVED_GRAPHS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


@app.get("/api/graph/diagnostics")
def live_graph_diagnostics():
    """Full diagnostics snapshot — superset of /api/graph/live/status."""
    running = _live_sim_thread is not None and _live_sim_thread.is_alive()
    sim_cache_size = len(_live_executor._sim_cache) if _live_executor is not None else 0
    return {"running": running, "sim_cache_entries": sim_cache_size, **_live_stats}


# ── Shootout — evolutionary generator (see image_pipeline/shootout/) ──
# NOTE: these static routes MUST stay registered before the dynamic
# /api/graph/{gid} routes below (FastAPI matches in registration order).

from image_pipeline.shootout import session as _shootout_session
from image_pipeline.shootout import store as _shootout_store
from image_pipeline.shootout import taste as _shootout_taste
from image_pipeline.shootout import config as _shootout_config


class ShootoutSessionRequest(BaseModel):
    session_id: str | None = None


class ShootoutRunRequest(BaseModel):
    session_id: str
    ratings: dict[str, int] | None = None  # /evolve: rate-if-not-already
    notes: dict[str, str] | None = None    # per-genome pros/cons free text
    node_feedback: dict[str, dict[str, str]] | None = None  # phase 3: {gid:{node_id:text}}


class ShootoutConfigRequest(BaseModel):
    overrides: dict[str, Any] | None = None
    reset: bool = False


@app.get("/api/shootout/config")
def shootout_get_config():
    """Tunable settings + current values for the /shootout settings menu."""
    return _shootout_config.config_info()


@app.post("/api/shootout/config")
def shootout_set_config(req: ShootoutConfigRequest):
    """Persist settings overrides (or reset to defaults). Applies to the
    next generation run — an in-flight render keeps its old config."""
    if req.reset:
        _shootout_config.reset_overrides()
    elif req.overrides:
        _shootout_config.save_overrides({
            **_shootout_config.load_overrides(), **req.overrides})
    return _shootout_config.config_info()


@app.post("/api/shootout/session")
def shootout_session(req: ShootoutSessionRequest):
    """Start a new shootout session, or resume one by id."""
    session = _shootout_session.start_session(
        req.session_id, _shootout_config.effective_config())
    return _shootout_session.session_state(session["session_id"])


@app.get("/api/shootout/sessions")
def shootout_sessions():
    return _shootout_store.list_sessions()


@app.get("/api/shootout/session/{session_id}")
def shootout_session_state(session_id: str):
    state = _shootout_session.session_state(session_id)
    if state is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return state


@app.post("/api/shootout/session/{session_id}/cancel")
def shootout_session_cancel(session_id: str):
    """Signal a running generation for this session to abort early.

    Safe to call when nothing is running. Returns whether a generation was
    in flight (and is now signalled to stop between batches)."""
    was_running = _shootout_session.cancel_session(session_id)
    return {"ok": True, "was_running": was_running}


@app.post("/api/shootout/session/{session_id}/reset")
def shootout_session_reset(session_id: str):
    """Cancel any running generation and delete the session + its genomes.

    The cross-session ratings dataset and taste model are preserved."""
    removed = _shootout_session.reset_session(session_id)
    return {"ok": True, "removed": removed}


def _shootout_launch_job(session_id: str) -> dict:
    """Run one generation in a background job; progress + result stream via
    the existing /api/jobs/{id}/stream SSE."""
    _evict_old_jobs()
    job_id = uuid.uuid4().hex[:8]
    job = {
        "id": job_id, "status": "running", "q": queue.Queue(),
        "output_path": None, "type": "shootout", "start": time.time(),
        "cancel_event": threading.Event(),
    }
    _jobs[job_id] = job

    def _worker():
        q = job["q"]
        try:
            result = _shootout_session.run_generation(
                session_id, _shootout_config.effective_config(),
                progress_cb=lambda m: q.put(("progress", m)))
            q.put(("done", result))
        except Exception as exc:
            q.put(("error", str(exc)))
        finally:
            job["status"] = "done"
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id, "session_id": session_id}


@app.post("/api/shootout/generate")
def shootout_generate(req: ShootoutRunRequest):
    """Generate + repair + render + reject one generation (async job)."""
    return _shootout_launch_job(req.session_id)


@app.post("/api/shootout/rate")
def shootout_rate(req: ShootoutRunRequest):
    """Persist star ratings and pros/cons notes for the latest generation."""
    if not req.ratings and not req.notes and not req.node_feedback:
        raise HTTPException(400, "ratings, notes, or node_feedback required")
    try:
        return _shootout_session.rate(req.session_id, req.ratings or {},
                                      _shootout_config.effective_config(),
                                      notes=req.notes,
                                      node_feedback=req.node_feedback)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/shootout/evolve")
def shootout_evolve(req: ShootoutRunRequest):
    """Rate/annotate (if included) then breed + render the next generation.
    Notes and per-node feedback are interpreted by the advisor and steer the
    breeding."""
    if req.ratings or req.notes or req.node_feedback:
        try:
            _shootout_session.rate(req.session_id, req.ratings or {},
                                   _shootout_config.effective_config(),
                                   notes=req.notes,
                                   node_feedback=req.node_feedback)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
    return _shootout_launch_job(req.session_id)


@app.get("/api/shootout/genome/{genome_id}")
def shootout_genome(genome_id: str):
    """Full genome envelope — graph is loadable in the normal editor."""
    genome = _shootout_store.load_genome(genome_id)
    if genome is None:
        raise HTTPException(404, f"Genome '{genome_id}' not found")
    return genome


@app.post("/api/shootout/train")
def shootout_train():
    """Retrain the taste model from the ratings dataset; returns metrics."""
    artifact = _shootout_taste.train()
    return {k: v for k, v in artifact.items()
            if k in ("trained", "n_samples", "metrics", "note", "model")}


@app.get("/api/shootout/suggest-ratings")
def shootout_suggest_ratings(k: int = 5):
    """Active-learning: surface the k most informative unrated-alive genomes.

    Returns a list of suggestion dicts (genome_id, fitness, novelty,
    temporal_var, n_nodes, reason) so the user can rate high-information-gain
    clips instead of random ones — directly attacks the starved-rating problem
    (Route 8, rating-signal poverty). See shootout/rating_suggest.py.
    """
    from image_pipeline.shootout.rating_suggest import suggest_for_rating
    k = max(1, min(int(k), 50))
    try:
        sug = suggest_for_rating(k=k, cfg=_shootout_config.effective_config())
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"suggest-ratings failed: {exc}")
    return {"suggestions": sug, "k": k}


@app.post("/api/shootout/contribution/{genome_id}")
def shootout_contribution(genome_id: str):
    """Per-node contribution diagnosis for one genome (node ablation).

    Renders a short baseline + one probe per node with that node
    bypassed/removed, and classifies each node: terminal, essential,
    contributes, silent (wired in but no visible effect), or disconnected
    (never reaches the output). The union silent+disconnected is the
    'not contributing' set the UI can surface on the graph.
    """
    from image_pipeline.shootout import contribution as _shootout_contrib
    from image_pipeline.shootout.generator import build_gene_pool
    genome = _shootout_store.load_genome(genome_id)
    if genome is None:
        raise HTTPException(404, f"Genome '{genome_id}' not found")
    cfg = _shootout_config.effective_config()
    pool = build_gene_pool(cfg)
    return _shootout_contrib.analyze_contribution(genome, cfg, pool)


@app.get("/api/shootout/render-status")
def shootout_render_status():
    """Live board of every clip currently rendering: which frame, the node
    cooking right now, sim sub-frame, and seconds elapsed / on the current
    frame. Drives the UI's live-render panel + skip buttons so a hang is
    visible and cancellable second-by-second."""
    from image_pipeline.shootout import progress as _shootout_progress
    import time as _t
    now = _t.time()
    # include_done=True so a just-finished genome KEEPS its captured
    # preview in the feed — the user sees the final still until the
    # next run's clear_all() wipes the board. Done genomes with
    # no preview (finished before frame 0 captured) are dropped.
    out = []
    for gid, s in _shootout_progress.MONITOR.snapshot(include_done=True).items():
        if s.get("done") and not s.get("preview"):
            continue
        out.append({
            "genome_id": gid,
            "frame": s.get("frame", 0),
            "total_frames": s.get("total_frames", 0),
            "n_nodes": s.get("n_nodes", 0),
            "node_id": s.get("node_id"),
            "node_name": s.get("node_name"),
            "node_method": s.get("node_method"),
            "sim_frame": s.get("sim_frame", 0),
            "elapsed_s": round(now - s.get("t0", now), 1),
            "frame_s": round(now - s.get("t_frame", now), 1),
            "skip_requested": bool(s.get("skip_requested")),
            # Live preview thumbnail (data: URL) — null until the first
            # captured frame. Lets the UI show a playless still so the
            # user can skip a candidate before it reaches the pool.
            "preview": s.get("preview"),
            "preview_frame": s.get("preview_frame", -1),
        })
    out.sort(key=lambda r: r["genome_id"])
    return {"rendering": out}


@app.post("/api/shootout/skip/{genome_id}")
def shootout_skip(genome_id: str):
    """Request an in-flight clip be skipped (skip button / hunting hangs).

    Sets the skip event the executor's sim loop polls and the render loop
    checks between frames; the clip is culled as 'skipped'. Returns
    active=False if that genome isn't currently rendering."""
    from image_pipeline.shootout import progress as _shootout_progress
    active = _shootout_progress.MONITOR.request_skip(genome_id)
    return {"ok": True, "genome_id": genome_id, "active": active}


@app.get("/api/shootout/utilization")
def shootout_utilization(session_id: str | None = None):
    """Gene-pool utilization audit (phase 2).

    With session_id: union of every genome rendered in that session.
    Without it: a fresh audit over one render_pool of random genomes, so
    the UI can show coverage even before any session has run.
    """
    from image_pipeline.shootout import utilization as _shootout_util
    from image_pipeline.shootout.generator import build_gene_pool
    from image_pipeline.shootout.repair import sample_valid_genome
    cfg = _shootout_config.effective_config()
    pool = build_gene_pool(cfg)
    genomes: list[dict] = []
    if session_id:
        s = _shootout_store.load_session(session_id)
        if s is not None:
            for gen in s.get("generations", []):
                for gid in gen.get("pool", []):
                    g = _shootout_store.load_genome(gid)
                    if g is not None:
                        genomes.append(g)
    if not genomes:
        import random as _random
        rng = _random.Random()
        genomes = [sample_valid_genome(pool, cfg, rng, origin="random")
                   for _ in range(cfg.render_pool)]
    return _shootout_util.audit_population(genomes, pool, cfg)


@app.get("/api/shootout/timeout-blame")
def shootout_timeout_blame():
    """Timeout blame — which methods/nodes waste the render budget.

    A clip culled as `timeout` (or `over-budget`) burned the full
    render budget only to be discarded. This aggregates the whole genome
    corpus: the per-method repeat offenders (the "problematic" set to
    target for debugging / speed work), per-clip attribution, and the
    worst (most expensive) clips. See shootout/timeout_blame.py.
    """
    from image_pipeline.shootout import timeout_blame as _tb
    cfg = _shootout_config.effective_config()
    rep = _tb.report(cfg)
    return rep


# ══════════════════════════════════════════════════════════════════════
# Tuning mode — directed brief → Hermes-built graph → critique → learned
# node-craft. The directed inverse of shootout. Endpoints are thin wrappers
# over image_pipeline.tuning.session. REGISTERED BEFORE the dynamic
# /api/graph/{gid} route below (FastAPI matches in registration order).
# ══════════════════════════════════════════════════════════════════════
from image_pipeline.tuning import session as _tune_session
from image_pipeline.tuning import store as _tune_store
from image_pipeline.tuning import catalog as _tune_catalog
from image_pipeline.tuning.hermes import hermes_available as _tune_hermes_ok
from image_pipeline.tuning.hermes import unavailable_message as _tune_hermes_msg
from image_pipeline.tuning.builder import is_motion_brief as _tune_is_motion


class TuneSessionRequest(BaseModel):
    session_id: str | None = None


class TuneBuildRequest(BaseModel):
    session_id: str | None = None
    brief: str = ""
    seed: int = 42
    width: int = 768
    height: int = 512


class TuneReviseRequest(BaseModel):
    session_id: str
    critique: str = ""
    seed: int = 42
    width: int = 768
    height: int = 512


class TuneRateRequest(BaseModel):
    session_id: str
    rating: int = 3
    critique: str = ""


def _tune_render_still(graph: dict, seed: int, width: int, height: int) -> str:
    """Kick off a single-frame render job for a graph; return its job_id.

    Reuses the existing job path (execute_graph) so the UI streams progress via
    /api/jobs/{id}/stream and fetches the PNG at /api/jobs/{id}/result."""
    req = GraphRequest(nodes=graph.get("nodes", []), edges=graph.get("edges", []),
                       seed=seed, frames=1, frame=0, width=width, height=height)
    return execute_graph(req)["job_id"]


def _tune_render_clip(graph: dict, session_id: str, frames: int = 48, fps: int = 24,
                      width: int = 768, height: int = 512) -> str | None:
    """Render a graph to a short mp4 (blocking) and return its video_url, or None.

    Mirrors the sequence worker: renders each frame via GraphExecutor and pipes to
    frames_to_mp4. Shared by the animate endpoint and the motion-brief auto-animate
    path in build/revise."""
    nodes = [dict(n) for n in graph.get("nodes", [])]
    edges = graph.get("edges", [])
    seed = graph.get("seed", 42)
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', f"tune-{session_id}")
    seq_dir = SEQUENCES_DIR / name
    work_dir = seq_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    from image_pipeline.core.utils import set_canvas as _set_canvas
    from image_pipeline.core.animation import frames_to_mp4 as _frames_to_mp4

    # Set the canvas BEFORE frames_to_mp4 — it reads W/H when building the ffmpeg
    # command, ahead of the first frame being pulled from the generator.
    _set_canvas(width, height)

    def _frame_gen():
        executor = GraphExecutor(work_dir, fps=fps, in_memory=True)
        for n in nodes:
            n["dirty"] = True
        for frame in range(frames):
            flat_outputs, terminal_id, _err = executor.execute(
                nodes, edges, seed, frame=frame, frames=frames)
            render_id = next((n["id"] for n in nodes if n.get("render")), None)
            if render_id and render_id in flat_outputs:
                terminal_id = render_id
            arr = (flat_outputs.get(terminal_id) or {}).get("image") if terminal_id else None
            if arr is not None:
                yield arr

    out = _frames_to_mp4(_frame_gen(), seq_dir / "output.mp4", fps=fps)
    if out is None:
        return None
    return f"/api/sequences/{name}/video.mp4?t={int(time.time())}"


def _tune_render_response(res: dict, brief: str, seed: int,
                          width: int, height: int) -> dict:
    """Build the build/revise response, auto-animating when the brief is about
    motion. Falls back to a still if the clip render fails."""
    base = {"ok": True, "session_id": res["session_id"],
            "attempt_id": res["attempt_id"], "graph": res["graph"],
            "rationale": res["rationale"],
            "graph_summary": _tune_catalog.describe_graph(res["graph"])}
    if _tune_is_motion(brief):
        try:
            url = _tune_render_clip(res["graph"], res["session_id"],
                                    width=width, height=height)
        except Exception:
            url = None
        if url:
            return {**base, "animated": True, "video_url": url}
    job_id = _tune_render_still(res["graph"], seed, width, height)
    return {**base, "job_id": job_id}


@app.post("/api/tune/session")
def tune_session(req: TuneSessionRequest):
    """Start a new tuning session, or resume one by id."""
    s = _tune_session.start(req.session_id)
    return {"session_id": s["session_id"],
            "current_brief": s.get("current_brief", ""),
            "playbook": _tune_store.read_playbook()}


@app.get("/api/tune/playbook")
def tune_playbook():
    """The accumulated node-craft playbook (the agent's growing understanding)."""
    return {"playbook": _tune_store.read_playbook()}


@app.get("/api/tune/session/{session_id}")
def tune_session_state(session_id: str):
    s = _tune_store.load_session(session_id)
    if s is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {"session_id": s["session_id"],
            "current_brief": s.get("current_brief", ""),
            "current_graph": s.get("current_graph"),
            "critique_history": s.get("critique_history", []),
            "attempts": len(s.get("attempts", []))}


@app.post("/api/tune/build")
def tune_build(req: TuneBuildRequest):
    """Hermes builds a graph from the brief, it is repaired to validity, and a
    still is rendered. Returns the graph, rationale, and the render job_id."""
    if not _tune_hermes_ok():
        return {"ok": False, "error": _tune_hermes_msg()}
    if not (req.brief or "").strip():
        raise HTTPException(400, "brief is required")

    res = _tune_session.build(req.session_id or "", req.brief)
    if not res["ok"]:
        return {"ok": False, "session_id": res.get("session_id"),
                "rationale": res.get("rationale", ""), "error": res["error"]}

    return _tune_render_response(res, req.brief, req.seed, req.width, req.height)


@app.post("/api/tune/revise")
def tune_revise(req: TuneReviseRequest):
    """Hermes revises the session's current graph given a critique, re-renders."""
    if not _tune_hermes_ok():
        return {"ok": False, "error": _tune_hermes_msg()}
    if not (req.critique or "").strip():
        raise HTTPException(400, "critique is required")

    res = _tune_session.revise(req.session_id, req.critique)
    if not res["ok"]:
        return {"ok": False, "session_id": req.session_id,
                "rationale": res.get("rationale", ""), "error": res["error"]}

    # Same motion → animate rule as build, keyed off the session's original brief.
    s = _tune_store.load_session(req.session_id) or {}
    brief = s.get("current_brief", "")
    return _tune_render_response(res, brief, req.seed, req.width, req.height)


@app.post("/api/tune/rate")
def tune_rate(req: TuneRateRequest):
    """Rate the current attempt; Hermes distills one durable lesson into the
    playbook. Returns the lesson and the refreshed playbook."""
    if not _tune_hermes_ok():
        return {"ok": False, "error": _tune_hermes_msg()}
    res = _tune_session.rate(req.session_id, req.rating, req.critique)
    if not res["ok"]:
        return {"ok": False, "error": res.get("error", "rate failed")}
    return {"ok": True, "section": res["section"], "lesson": res["lesson"],
            "written": res["written"], "playbook": _tune_store.read_playbook()}


@app.post("/api/tune/animate")
def tune_animate(req: TuneSessionRequest, frames: int = 48, fps: int = 24,
                 width: int = 768, height: int = 512):
    """Render the session's current graph as a short mp4 (blocking).

    Renders each frame via GraphExecutor (mirroring the sequence worker) and
    pipes to frames_to_mp4. Served from the standard sequences path."""
    s = _tune_store.load_session(req.session_id or "")
    if s is None or not s.get("current_graph"):
        return {"ok": False, "error": "no current graph to animate"}

    try:
        url = _tune_render_clip(s["current_graph"], req.session_id or "",
                                frames=frames, fps=fps, width=width, height=height)
    except Exception as exc:
        return {"ok": False, "error": f"animate failed: {exc}"}
    if url is None:
        return {"ok": False, "error": "no frames rendered"}
    return {"ok": True, "video_url": url}


@app.get("/api/graph/{gid}")
def get_graph(gid: str = "active"):
    """Return the shared graph document (nodes, edges, canvas, meta)."""
    with _graph_store_lock:
        doc = _load_graph_doc(gid)
        return json.loads(json.dumps(doc))  # copy


@app.put("/api/graph/{gid}")
async def put_graph(gid: str, payload: dict, by: str | None = None):
    """Replace the whole graph (wholesale edit). Persists + broadcasts."""
    with _graph_store_lock:
        doc = _graph_default()
        doc["id"] = gid
        doc["nodes"] = payload.get("nodes", [])
        doc["edges"] = payload.get("edges", [])
        if "canvas" in payload:
            doc["canvas"] = payload["canvas"]
        _touch_graph_meta(doc, by or "agent")
        _persist_graph_doc(doc)
    _broadcast_graph_event("graph:replace", {"gid": gid, "doc": doc, "by": by or "agent"})
    return {"ok": True, "id": gid}


@app.patch("/api/graph/{gid}")
async def patch_graph(gid: str, patch: GraphPatch):
    """Apply fine-grained ops: add_node, update_node, remove_node,
    add_edge, remove_edge, set_canvas, clear. Origin-agnostic."""
    with _graph_store_lock:
        doc = _load_graph_doc(gid)
        nodes = doc["nodes"]
        edges = doc["edges"]
        by = patch.by or "agent"
        applied = []
        for op in patch.ops:
            kind = op.get("op")
            if kind == "add_node":
                node = op.get("node", {})
                if "id" not in node:
                    node["id"] = op.get("id") or f"n{int(time.monotonic()*1000)}"
                nodes.append(node)
                applied.append(kind)
            elif kind == "update_node":
                nid = op.get("id")
                for n in nodes:
                    if n["id"] == nid:
                        n.update(op.get("params", {}))
                        if "method_id" in op:
                            n["method_id"] = op["method_id"]
                        break
                applied.append(kind)
            elif kind == "remove_node":
                nid = op.get("id")
                doc["nodes"] = [n for n in nodes if n["id"] != nid]
                doc["edges"] = [e for e in edges
                                if e.get("src_node") != nid and e.get("dst_node") != nid]
                applied.append(kind)
            elif kind == "add_edge":
                edges.append(op.get("edge", {}))
                applied.append(kind)
            elif kind == "remove_edge":
                eid = op.get("id")
                doc["edges"] = [e for e in edges if e.get("id") != eid]
                applied.append(kind)
            elif kind == "set_canvas":
                doc["canvas"] = op.get("canvas", doc["canvas"])
                applied.append(kind)
            elif kind == "clear":
                doc["nodes"] = []
                doc["edges"] = []
                applied.append(kind)
        _touch_graph_meta(doc, by)
        _persist_graph_doc(doc)
    _broadcast_graph_event("graph:patch", {"gid": gid, "ops": patch.ops, "by": by, "doc": doc})
    return {"ok": True, "id": gid, "applied": applied}


@app.websocket("/api/graph/ws")
async def graph_ws(websocket: WebSocket):
    """Subscribe to graph mutations (user + agent edits). Sends the current
    doc on connect, then graph:* events as edits arrive."""
    await websocket.accept()
    with _graph_store_lock:
        doc = _load_graph_doc("active")
        current = json.loads(json.dumps(doc))
    try:
        await websocket.send_text(json.dumps({"event": "graph:state", "doc": current}))
    except Exception:
        pass
    with _GRAPH_WS_LOCK:
        _GRAPH_WS_CLIENTS.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _GRAPH_WS_LOCK:
            _GRAPH_WS_CLIENTS.discard(websocket)


# Defined before its first endpoint use — pydantic resolves the annotation
# at route-registration time, so a later definition breaks module import on
# eager-resolving pydantic versions.
class GraphRequest(BaseModel):
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seed: int = 42
    frames: int = 1
    frame: int = 0
    width: int = 768
    height: int = 512
    graph_id: str | None = None  # optional shared-doc id; if set, live loop reads this doc each frame


@app.post("/api/graph/{gid}/execute")
async def execute_graph_by_id(gid: str, req: GraphRequest | None = None):
    """Execute the shared graph (or an override request) headlessly and
    return a job. For programmatic/agent use this avoids re-posting the full
    graph each time — the shared doc is the source of truth."""
    with _graph_store_lock:
        doc = _load_graph_doc(gid)
    if req is not None and (req.nodes or req.edges):
        nodes, edges = req.nodes, req.edges
        width, height = req.width, req.height
    else:
        # Use the shared doc, but allow execute-time width/height override
        nodes, edges = doc["nodes"], doc["edges"]
        width = (req.width if req is not None else 0) or doc["canvas"].get("w", 768)
        height = (req.height if req is not None else 0) or doc["canvas"].get("h", 512)
    # Mirror the doc into the live request shape so execute_graph runs it
    exec_req = GraphRequest(
        nodes=nodes, edges=edges, seed=42,
        frames=req.frames if req else 1,
        frame=req.frame if req else 0,
        width=width, height=height,
    )
    return execute_graph(exec_req)


# 3D node IDs derived from the authoritative defs dict — stays in sync automatically
_CLIENT_3D_IDS = frozenset(_THREEJS_3D_NODE_DEFS.keys())

_THREEJS_SIDECAR_URL = os.environ.get(
    'THREEJS_SIDECAR_URL', 'http://127.0.0.1:7862'
)


@app.get("/api/graph/{gid}/render")
def render_graph_bytes(gid: str = "active", frame: int = 0, fmt: str = "png",
                       width: int = 0, height: int = 0):
    """Synchronous headless render → raw image bytes (no base64, no job poll).
    For agent/programmatic use. Reads the shared doc; width/height override the
    doc's canvas. fmt=png (default) or jpg (quality param).

    If the graph contains three.js 3D nodes, proxies to the Node.js sidecar
    (image_pipeline/3d/threejs-sidecar.mjs) for headless WebGL rendering."""
    import numpy as np
    from PIL import Image
    from fastapi import HTTPException
    from image_pipeline.core.utils import set_canvas as _set_canvas
    with _graph_store_lock:
        doc = _load_graph_doc(gid)
    nodes = [dict(n) for n in doc["nodes"]]
    edges = [dict(e) for e in doc["edges"]]
    cw = width or doc["canvas"].get("w", 768)
    ch = height or doc["canvas"].get("h", 512)
    if not nodes:
        raise HTTPException(400, "Graph is empty")

    # Proxy 3D graphs to Node.js sidecar
    has_3d = any(n.get("method_id") in _CLIENT_3D_IDS for n in nodes)
    if has_3d:
        import httpx
        try:
            resp = httpx.post(
                f"{_THREEJS_SIDECAR_URL}/render",
                json={"nodes": nodes, "edges": edges,
                       "width": cw, "height": ch, "frame": frame},
                timeout=30,
            )
            if resp.status_code == 200:
                return Response(
                    content=resp.content,
                    media_type=resp.headers.get("content-type", "image/png"),
                    headers={"X-Render-Ms": resp.headers.get("x-render-ms", "")},
                )
            raise HTTPException(502, f"3D sidecar returned {resp.status_code}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"3D sidecar unavailable: {e}")

    # Persistent shared executor, serialized by a lock. Sync FastAPI endpoints
    # run on a threadpool — a thread-local executor here meant up to N
    # executors × unbounded sim caches, and cache hits were a per-thread
    # lottery. One executor + lock gives every call the warm cache.
    with _render_exec_lock:
        _re = _ensure_executor(
            _render_exec_state.get('executor'),
            _render_exec_state.get('last_nodes'),
            _render_exec_state.get('last_edges'),
            _render_exec_state.get('last_gid', ''),
            _render_exec_state.get('last_seed', 0),
            _render_exec_state.get('last_canvas', (0, 0)),
            gid, 42, (cw, ch), nodes, edges,
            copy_nodes=False,
        )
        (_render_exec_state['executor'], _render_exec_state['last_nodes'],
         _render_exec_state['last_edges'], _render_exec_state['last_gid'],
         _render_exec_state['last_seed'], _render_exec_state['last_canvas']) = _re
        _set_canvas(cw, ch)
        ex = _render_exec_state['executor']
        flat, terminal_id, errs = ex.execute(nodes, edges, 42, frame=frame, frames=1)
    render_id = next((n["id"] for n in nodes if n.get("render")), None)
    if render_id and render_id in flat:
        terminal_id = render_id
    if terminal_id is None:
        raise HTTPException(400, "No output node")
    arr = (flat.get(terminal_id) or {}).get("image")
    if arr is None:
        raise HTTPException(500, "Render produced no image")
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    if fmt == "jpg":
        img.save(buf, format="JPEG", quality=85)
        mt = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mt = "image/png"
    return Response(content=buf.getvalue(), media_type=mt)


# ── Asset uploads (USD/GLTF models, textures) ─────────────────────────
# Raw-body upload (no python-multipart dependency): the client PUTs the file
# bytes with ?name=<filename>; the file is served back at /assets/<name>.

_ASSET_MAX_BYTES = 512 * 1024 * 1024  # 512 MB — local tool, generous cap


@app.post("/api/assets/upload")
async def upload_asset(request: Request, name: str = "asset.bin"):
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', Path(name).name) or "asset.bin"
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "Empty upload body")
    if len(raw) > _ASSET_MAX_BYTES:
        raise HTTPException(413, f"Asset exceeds {_ASSET_MAX_BYTES // (1024*1024)} MB cap")
    path = ASSETS_DIR / safe
    if path.exists() and path.read_bytes() != raw:
        # Same name, different content — version the filename instead of clobbering.
        stem, suffix = path.stem, path.suffix
        i = 1
        while (ASSETS_DIR / f"{stem}-{i}{suffix}").exists():
            i += 1
        path = ASSETS_DIR / f"{stem}-{i}{suffix}"
    path.write_bytes(raw)
    return {"ok": True, "url": f"/assets/{path.name}", "name": path.name, "bytes": len(raw)}


@app.get("/api/assets")
def list_assets():
    out = []
    for p in sorted(ASSETS_DIR.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            out.append({"name": p.name, "url": f"/assets/{p.name}", "bytes": p.stat().st_size})
    return out


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
        "saved_at": datetime.now(timezone.utc).isoformat(),
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
# Stats updated by the live loop — read by /api/graph/live/status and /api/graph/diagnostics
_live_stats: dict = {
    "frame": 0, "cook_ms": 0.0, "fps": 0.0,
    # per-node timing
    "node_timings": {}, "node_names": {},
    # per-node errors (node_id → short error string)
    "node_errors": {},
    # cache
    "cache_hits": 0, "cache_misses": 0,
    "total_cache_hits": 0, "total_cache_misses": 0,
    "last_invalidated": 0,
    # data flow
    "mem_edges": 0, "disk_edges": 0,
    "gpu_nodes": 0, "cpu_nodes": 0,
    # overhead split
    "node_compute_ms": 0.0, "overhead_ms": 0.0,
    # graph topology
    "active_nodes": 0, "active_edges": 0,
    "canvas_w": 0, "canvas_h": 0,
}

# Persistent executor — survives hot-swaps so Arch-A sim caches are kept
_live_executor = None          # GraphExecutor | None
_live_last_nodes: list = []
_live_last_edges: list = []
_live_last_gid:   str  = ""
_live_last_seed:  int  = 0
_live_last_canvas: tuple = (0, 0)

# Shared render executor for /api/graph/{gid}/render — one executor, one
# warm cache, serialized by the lock (the executor is not thread-safe).
_render_exec_state: dict = {}
_render_exec_lock = threading.Lock()


def _ensure_executor(
    executor, last_nodes, last_edges, last_gid, last_seed, last_canvas,
    current_gid, current_seed, canvas, nodes, edges, *,
    session_dir=_GRAPH_SESSION_DIR, copy_nodes=True, stats=None,
    audit_to_disk=True,
) -> tuple:
    """Manage a persistent GraphExecutor with gid-based identity and seed-based flush.

    Args:
        executor: current GraphExecutor or None
        last_nodes/edges/gid/seed/canvas: prior-frame state trackers
        current_gid/seed/canvas: this frame's identity & params
        nodes/edges: this frame's graph data
        session_dir: root for Arch-A sim cache files
        copy_nodes: if True, make fresh [dict(n) for n in nodes] copies for storage
        stats: optional dict to populate with invalidation info

    Returns (executor, last_nodes, last_edges, last_gid, last_seed, last_canvas).
    Caller assigns the tuple back to its state.
    """
    if executor is None or canvas != last_canvas or current_gid != last_gid:
        executor = GraphExecutor(session_dir, in_memory=True,
                                 audit_to_disk=audit_to_disk)
        last_canvas = canvas
        last_gid = current_gid
        last_seed = current_seed
        last_nodes = None
        last_edges = None
        if stats is not None:
            stats["cache_flush_reason"] = "new_executor"
            stats["last_invalidated"] = 0
            stats.setdefault("total_cache_hits", 0)
            stats.setdefault("total_cache_misses", 0)
    elif current_seed != last_seed:
        executor._sim_cache.clear()
        executor._sim_params_hash.clear()
        last_seed = current_seed
        last_nodes = None
        last_edges = None
        if stats is not None:
            stats["cache_flush_reason"] = "seed_changed"
            stats["last_invalidated"] = 0
    else:
        inv_count = executor.selective_invalidate(
            last_nodes or [], nodes, last_edges or [], edges, current_seed,
        )
        last_nodes = [dict(n) for n in nodes] if copy_nodes else nodes
        last_edges = [dict(e) for e in edges] if copy_nodes else edges
        if stats is not None:
            stats["cache_flush_reason"] = "selective"
            stats["last_invalidated"] = inv_count
        return executor, last_nodes, last_edges, last_gid, last_seed, last_canvas

    # New executor or full flush — store clean snapshots
    last_nodes = [dict(n) for n in nodes] if copy_nodes else nodes
    last_edges = [dict(e) for e in edges] if copy_nodes else edges
    return executor, last_nodes, last_edges, last_gid, last_seed, last_canvas


@app.post("/api/graph/live")
def live_graph_sim(req: GraphRequest):
    """Start or stop a continuous graph simulation.

    frames == 0 stops; anything else (re)starts. Only one live loop runs at
    a time — starting while one is active stops the old loop first, so
    re-POSTing with an edited graph hot-swaps it.

    The GraphExecutor is kept alive across hot-swaps. Arch-A simulation caches
    survive unchanged hot-swaps; only nodes with changed non-volatile params
    are invalidated.
    """
    global _live_sim_cancel, _live_sim_thread
    global _live_executor, _live_last_nodes, _live_last_edges, _live_last_gid, _live_last_seed, _live_last_canvas
    with _live_sim_lock:
        # Stop any existing loop (both for stop requests and restarts)
        if _live_sim_thread is not None and _live_sim_thread.is_alive():
            _live_sim_cancel.set()
            _live_sim_thread.join(timeout=5.0)
        _live_sim_thread = None
        if req.frames == 0:
            return {"status": "stopped"}

        cancel = threading.Event()
        # Graph source of truth: the shared doc. The live loop reads it every
        # frame, so a clear (user or agent) stops the render. The client POSTs
        # the current graph in the request body on every start/swap (it does
        # not PUT the doc itself), so the body is authoritative for client
        # edits — persist it into the doc so the loop re-reads the latest
        # graph each frame. A cleared (empty) doc still breaks the loop.
        gid = req.graph_id if getattr(req, "graph_id", None) else "active"
        if req.nodes:
            # Client is driving: seed/overwrite the shared doc with the body
            # graph so the loop has a non-empty, up-to-date source of truth.
            with _graph_store_lock:
                doc = _graph_default()
                doc["id"] = gid
                doc["nodes"] = req.nodes
                doc["edges"] = req.edges
                if req.width and req.height:
                    doc["canvas"] = {"w": req.width, "h": req.height}
                _touch_graph_meta(doc, "live-start")
                _persist_graph_doc(doc)
            nodes, edges = req.nodes, req.edges
        else:
            with _graph_store_lock:
                _gdoc = _load_graph_doc(gid)
            nodes, edges = _gdoc["nodes"], _gdoc["edges"]
        seed = req.seed
        width, height = req.width, req.height

        # ── Persistent executor: reuse or create ──────────────────────
        _live_executor, _live_last_nodes, _live_last_edges, \
            _live_last_gid, _live_last_seed, _live_last_canvas = \
            _ensure_executor(
                _live_executor, _live_last_nodes, _live_last_edges,
                _live_last_gid, _live_last_seed, _live_last_canvas,
                gid, seed, (width, height),
                nodes, edges,
                session_dir=OUTPUT_ROOT / "_live_sim",
                copy_nodes=True,
                stats=_live_stats,
                # Live mode is memory-only transport — nothing reads the
                # _live_sim dir, so skip every per-node PNG/sidecar write.
                audit_to_disk=False,
            )
        executor = _live_executor
        reason = _live_stats.get("cache_flush_reason", "unknown")
        inv = _live_stats.get("last_invalidated", 0)
        if reason == "new_executor":
            _inv_msg = "new executor"
        elif reason == "seed_changed":
            _inv_msg = "seed changed — full flush"
        else:
            _inv_msg = f"selective invalidation: {inv} entries cleared"

        def _live_loop():
            from image_pipeline.core.utils import set_canvas as _set_canvas
            from image_pipeline.core.registry import get_meta as _get_meta_ll
            from image_pipeline.core.graph import _compute_live_dirty as _cld
            _set_canvas(req.width, req.height)
            live_w, live_h = req.width, req.height
            frame = 0
            consecutive_errors = 0
            # Params snapshot for per-node change detection (excludes volatile keys)
            _last_params: dict[str, dict] = {}
            LIVE_TOTAL_FRAMES = 300
            _frame_interval = 1.0 / 30.0
            print(f"[live-sim] starting loop, {len(nodes)} nodes, {len(edges)} edges, {_inv_msg}")
            while not cancel.is_set():
                # ── Read the shared graph doc every frame (single source of
                # truth). A clear (user or agent) empties it → stop the render.
                with _graph_store_lock:
                    _doc = _load_graph_doc(gid)
                    if not _doc["nodes"]:
                        print("[live-sim] shared graph is empty — stopping loop")
                        break
                    # Fresh working copy so in-place time injection doesn't
                    # mutate the persisted doc.
                    work_nodes = [dict(n) for n in _doc["nodes"]]
                    work_edges = [dict(e) for e in _doc["edges"]]
                    cw = _doc["canvas"].get("w", live_w)
                    ch = _doc["canvas"].get("h", live_h)
                if (cw, ch) != (live_w, live_h):
                    live_w, live_h = cw, ch
                    _set_canvas(live_w, live_h)
                _tick_start = time.monotonic()
                try:
                    # ── Phase 6: Selective dirty marking ──────────────────────
                    # Build the initial set of dirty nodes for this frame:
                    #   • Time-varying nodes (is_time_varying=True) — always
                    #   • Arch-A simulation nodes — always (their output changes
                    #     each frame via the sim cache, frame index advances)
                    #   • Nodes whose user-facing params changed since last frame
                    #   • Nodes that have paramKeyframes (keyframe eval is frame-dependent)
                    # Then cascade that set forward through the DAG.
                    initially_dirty: set[str] = set()
                    for n in work_nodes:
                        nid = n["id"]
                        if "params" not in n:
                            n["params"] = {}

                        meta = _get_meta_ll(n.get("method_id", ""))
                        is_tv = True if meta is None else meta.is_time_varying
                        # Arch-A nodes always re-cook (frame index advances in sim cache)
                        from image_pipeline.core.arch import detect_architecture as _det_arch
                        if meta is not None and _det_arch(meta) == "A":
                            is_tv = True

                        if is_tv:
                            # Inject time only into time-varying nodes.
                            # Static nodes keep their last `time` value (or none),
                            # so their params hash stays stable.
                            n["params"]["time"] = float(frame)
                            initially_dirty.add(nid)
                        else:
                            # Non-time-varying: check if user params changed
                            cur = {k: v for k, v in n["params"].items()
                                   if k not in ("time", "frame", "frame_seed",
                                                "_timeline", "_input_image", "input_image")}
                            if cur != _last_params.get(nid):
                                initially_dirty.add(nid)
                            _last_params[nid] = dict(cur)

                        # Any node with active paramKeyframes is frame-dependent
                        if n.get("paramKeyframes"):
                            initially_dirty.add(nid)

                    # Cascade dirty forward through the topology
                    dirty_set = _cld(work_nodes, work_edges, initially_dirty)
                    for n in work_nodes:
                        n["dirty"] = n["id"] in dirty_set

                    flat_outputs, terminal_id, node_errors = executor.execute(
                        work_nodes, work_edges, seed, frame=frame % LIVE_TOTAL_FRAMES, frames=LIVE_TOTAL_FRAMES
                    )
                    for nid, err in node_errors.items():
                        print(f"[live-sim] node {nid} error:\n{err}")
                    # Use the per-frame doc snapshot, not the (stale) request
                    # payload — render-flag/name edits mid-live apply next frame.
                    render_id = next((n["id"] for n in work_nodes if n.get("render")), None)
                    if render_id and render_id in flat_outputs:
                        terminal_id = render_id
                    arr = (flat_outputs.get(terminal_id) or {}).get("image") if terminal_id else None

                    # ── Update _live_stats BEFORE pushing the frame so the
                    #    WS message carries metadata for THIS frame. ──
                    frame += 1
                    consecutive_errors = 0
                    _cook_ms = (time.monotonic() - _tick_start) * 1000.0
                    _live_stats["frame"]    = frame
                    _live_stats["cook_ms"]  = round(_cook_ms, 1)
                    _live_stats["fps"]      = round(1000.0 / max(_cook_ms, 1.0), 1)
                    _live_stats["node_errors"] = {
                        nid: str(err).splitlines()[0][:120]
                        for nid, err in node_errors.items()
                    }
                    # Pull per-frame diagnostics from the executor
                    _fs = executor.last_frame_stats
                    if _fs:
                        _live_stats["node_timings"]   = _fs.get("node_timings", {})
                        _live_stats["cache_hits"]     = _fs.get("cache_hits", 0)
                        _live_stats["cache_misses"]   = _fs.get("cache_misses", 0)
                        _live_stats["total_cache_hits"]   += _fs.get("cache_hits", 0)
                        _live_stats["total_cache_misses"] += _fs.get("cache_misses", 0)
                        _live_stats["mem_edges"]      = _fs.get("mem_edges", 0)
                        _live_stats["disk_edges"]     = _fs.get("disk_edges", 0)
                        _live_stats["edge_transport"] = _fs.get("edge_transport", {})
                        _live_stats["gpu_nodes"]      = _fs.get("gpu_nodes", 0)
                        _live_stats["cpu_nodes"]      = _fs.get("cpu_nodes", 0)
                        _live_stats["node_compute_ms"]= _fs.get("node_compute_ms", 0.0)
                        _live_stats["overhead_ms"]    = _fs.get("overhead_ms", 0.0)
                        _live_stats["nodes_cooked"]   = _fs.get("nodes_cooked", 0)
                        _live_stats["nodes_skipped"]  = _fs.get("nodes_skipped", 0)
                    # Build node_names map from the current node list
                    from image_pipeline.core.registry import get_meta as _get_meta
                    _live_stats["node_names"] = {
                        n["id"]: (_get_meta(n.get("method_id","")) or type("M",[],{"name":n.get("method_id","?")})()).name
                        for n in work_nodes
                    }
                    _live_stats["active_nodes"] = len(work_nodes)
                    _live_stats["active_edges"] = len(work_edges)
                    _live_stats["canvas_w"]     = live_w
                    _live_stats["canvas_h"]     = live_h

                    # Push frame + per-frame metadata to MJPEG buffer and WS clients
                    if arr is not None:
                        _push_live_frame(arr, ws_meta=dict(_live_stats))
                except Exception:
                    import traceback as _tb
                    print(f"[live-sim] frame {frame} failed:\n{_tb.format_exc(limit=6)}")
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("[live-sim] 10 consecutive failures — stopping loop")
                        break
                    time.sleep(0.5)
                # Throttle to ~30fps so the browser can display each frame
                _elapsed = time.monotonic() - _tick_start
                _sleep = _frame_interval - _elapsed
                if _sleep > 0:
                    time.sleep(_sleep)

        _live_sim_cancel = cancel
        _live_sim_thread = threading.Thread(target=_live_loop, daemon=True)
        _live_sim_thread.start()
        return {"status": "running"}


@app.get("/api/graph/live/status")
def live_graph_status():
    running = _live_sim_thread is not None and _live_sim_thread.is_alive()
    return {"running": running, **_live_stats}


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
        # Include node params in the context hash so param changes invalidate
        # the single-frame cache. Without this, changing a param and hitting
        # Auto mode returns the cached frame from the previous run.
        _nodes_ctx = json.dumps(
            [{n["id"]: n.get("params", {})} for n in nodes],
            sort_keys=True, default=str
        )
        _run_ctx = (seed, start_frame, _nodes_ctx, json.dumps(edges, sort_keys=True, default=str))
        _ctx_changed = _last_single_frame_ctx != _run_ctx
        if n_frames > 1 or _ctx_changed:
            for n in nodes:
                n["dirty"] = True
        _last_single_frame_ctx = _run_ctx if n_frames == 1 else None
        print(f"[run-job] n_frames={n_frames}, ctx_changed={_ctx_changed}, dirty_flags={[n.get('dirty') for n in nodes]}")

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


def _anim_params_to_keyframes(anim_params: dict, start_frame: int, end_frame: int) -> dict:
    """Fold legacy per-node ``animParams`` (linear from→to tweens) into the
    single ``paramKeyframes`` model so every param animation runs through one
    engine (KeyframeTrack in core.graph).

    Each ``animParams[key] = {enabled, from, to}`` becomes a 2-keyframe linear
    track over [start_frame, end_frame], which evaluates identically to the old
    server-side linear tween.
    """
    tracks: dict[str, list[dict]] = {}
    if not anim_params:
        return tracks
    for key, anim in anim_params.items():
        if not isinstance(anim, dict) or not anim.get("enabled"):
            continue
        if "from" not in anim or "to" not in anim:
            continue
        tracks[key] = [
            {"frame": start_frame, "value": anim["from"], "easing": "linear"},
            {"frame": end_frame, "value": anim["to"], "easing": "linear"},
        ]
    return tracks


def _merge_anim_params_into_nodes(nodes: list[dict], start_frame: int, end_frame: int) -> None:
    """In-place: convert each node's ``animParams`` into ``paramKeyframes`` so
    the executor evaluates them via the shared keyframe engine. Idempotent with
    respect to ``paramKeyframes`` already present on the node (animParams tracks
    are added only for keys not already keyframed)."""
    for n in nodes:
        anim = n.get("animParams") or {}
        if not anim:
            continue
        pk = n.get("paramKeyframes") or {}
        folded = _anim_params_to_keyframes(anim, start_frame, end_frame)
        for k, track in folded.items():
            if k not in pk:
                pk[k] = track
        n["paramKeyframes"] = pk
        # animParams is now only sugar over paramKeyframes; drop to avoid the
        # executor seeing a stale, redundant mechanism.
        n.pop("animParams", None)


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

        # Fold legacy per-node animParams into the single paramKeyframes model
        # so every param animation is evaluated by one engine (KeyframeTrack)
        # inside the executor. This replaces the old server-side _interpolate_params
        # per-frame rebuild loop.
        _merge_anim_params_into_nodes(nodes, start_frame, end_frame)

        # Force all nodes dirty — the executor's dirty-flag skip reads cached
        # PNGs from disk, which would return the same frame 1 image for every
        # subsequent frame.
        for n in nodes:
            n["dirty"] = True

        for frame in range(start_frame, end_frame + 1):
            try:
                flat_outputs, terminal_id, frame_errors = executor.execute(
                    nodes, edges, seed, frame=frame, frames=n_frames
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


_ND_RUNNER = Path(__file__).resolve().parent / "nd_runner.py"

# ── Hermes agent — the sole LLM backend for all LLM calls ────────────
# The install location is configurable so Node Doctor works on any machine
# with a Hermes install, not just the one where it lives at the default path.
#   HERMES_AGENT_DIR — hermes-agent checkout (default ~/.hermes/hermes-agent)
#   HERMES_PYTHON    — interpreter to run nd_runner.py under
#                      (default <HERMES_AGENT_DIR>/venv/bin/python)
# nd_runner.py resolves the same variables, so both processes agree.
HERMES_AGENT_DIR = Path(
    os.environ.get("HERMES_AGENT_DIR", str(Path.home() / ".hermes" / "hermes-agent"))
)
_HERMES_PY = Path(
    os.environ.get("HERMES_PYTHON", str(HERMES_AGENT_DIR / "venv" / "bin" / "python"))
)

if _HERMES_PY.exists():
    print(f"[node-doctor] Hermes backend found: {_HERMES_PY}")
else:
    print(
        f"[node-doctor] WARNING: Hermes backend not found at {_HERMES_PY} — "
        f"Node Doctor chat will fail. Set HERMES_AGENT_DIR (or HERMES_PYTHON) "
        f"to your hermes-agent install."
    )


# ── Shared Hermes streaming helper (used by the Node Doctor chat) ────────
async def _nd_stream(system: str, messages: list, timeout: int = 120):
    """Stream a Node Doctor agent run as SSE text events."""
    if not _HERMES_PY.exists():
        msg = (f"⚠ Hermes backend not found at {_HERMES_PY}. "
               f"Set HERMES_AGENT_DIR (or HERMES_PYTHON) and restart.")
        yield f"data: {json.dumps({'text': msg})}\n\n"
        return

    stdin_bytes = json.dumps({"system_prompt": system, "messages": messages}).encode()
    try:
        proc = await asyncio.create_subprocess_exec(
            str(_HERMES_PY), str(_ND_RUNNER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_bytes), timeout=timeout
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


def _nd_system_for(method_id: str, node_def: dict, node_params: dict,
                   extra: str = "") -> str:
    """Build the node-context system prompt shared by the Node Doctor chat."""
    source = ""
    path = _get_method_path(method_id)
    if path and path.exists():
        source = path.read_text()
    return f"""{extra}
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


@app.post("/api/node-doctor/chat")
async def nd_chat(payload: dict):
    method_id   = payload.get("method_id", "")
    node_def    = payload.get("node_def", {})
    node_params = payload.get("node_params", {})
    messages    = payload.get("messages", [])

    system = _nd_system_for(method_id, node_def, node_params, _ND_SYSTEM)

    return StreamingResponse(
        _nd_stream(system, messages), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/node-doctor/apply", dependencies=[Depends(require_token)])
async def nd_apply(payload: dict):
    method_id  = payload.get("method_id", "")
    new_source = payload.get("source", "")
    if not new_source:
        return {"error": "No source provided"}

    path = _get_method_path(method_id)
    if not path or not path.exists():
        return {"error": "Method source file not found"}

    backup_id = uuid.uuid4().hex[:8]
    # Backups live outside methods/ — in-tree copies carry duplicate
    # @method ids, feed the file watcher, and pollute the audit scan.
    backup_dir = OUTPUT_ROOT / "nd-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.stem}.nd-bak-{backup_id}.py"
    shutil.copy2(str(path), str(backup_path))
    _nd_backups[backup_id] = (str(path), str(backup_path))

    path.write_text(new_source)
    # watchdog picks up the change and hot-reloads automatically
    return {"ok": True, "backup_id": backup_id}


@app.post("/api/node-doctor/undo/{backup_id}", dependencies=[Depends(require_token)])
async def nd_undo(backup_id: str):
    entry = _nd_backups.pop(backup_id, None)
    if not entry:
        return {"error": "Backup not found"}
    orig_path, backup_path = entry
    shutil.copy2(backup_path, orig_path)
    Path(backup_path).unlink(missing_ok=True)
    return {"ok": True}


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


@app.post("/api/node-tester/batch-apply", dependencies=[Depends(require_token)])
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
        if not API_TOKEN:
            print(
                "⚠  WARNING: tunneling without GRILLMASTER_API_TOKEN set — "
                "anyone with the URL can write method source and restart the "
                "server. Set the env var and put the token in the UI's "
                "localStorage['api-token']."
            )
        from pyngrok import ngrok
        tunnel = ngrok.connect(args.port, bind_tls=True)
        print(f"🌐 Public URL: {tunnel.public_url}")

    uvicorn.run("image_pipeline.server:app", host="0.0.0.0", port=args.port, reload=False, log_config=None)
