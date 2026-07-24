"""FastAPI server for the image-generation pipeline GUI."""
from __future__ import annotations
import asyncio
import base64
import glob
import importlib
import inspect
import io
import ast
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
       "fps_limit": bool, "target_fps": float,
       "img": "<base64-jpeg>"}

    Note `fps` is cook throughput (1/cook_ms), not the delivered rate — when
    fps_limit is on, frames arrive at target_fps.

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
            "fps_limit":    ws_meta.get("fps_limit", False),
            "target_fps":   ws_meta.get("target_fps", 0.0),
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
# reachable beyond localhost, set GRILLMASTER_API_TOKEN; those endpoints then
# require the X-Api-Token header. The UI attaches it automatically from
# localStorage['api-token'].
API_TOKEN = os.environ.get("GRILLMASTER_API_TOKEN", "")


def require_token(request: Request):
    if API_TOKEN and request.headers.get("x-api-token") != API_TOKEN:
        raise HTTPException(401, "Missing or invalid X-Api-Token header")


class _RevalidatingStatic(StaticFiles):
    """Serve UI assets with revalidation forced.

    The editor's JS/CSS are edited in place while the server keeps running, and
    a browser memory-cache hit silently serves the previous copy — which turns a
    saved fix into a phantom bug that survives a reload. ``no-cache`` still
    permits a 304 via the ETag, so this costs a conditional request per file,
    not a re-download.
    """

    def file_response(self, *args, **kwargs) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app = FastAPI(title="Image Pipeline", lifespan=lifespan)
app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")
# Static UI assets (vendored three.js, client-side executor modules). Additive —
# purely for serving front-end files; does not touch the render/export pipeline.
app.mount("/ui", _RevalidatingStatic(directory=str(UI_DIR)), name="ui")
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


def _derive_anim_mode_choices(fn) -> list[str] | None:
    """Best-effort: recover the enum of animation modes a method actually
    implements, by reading its own source.

    Many methods alias ``anim_mode`` to a local variable (e.g.
    ``mode = params.get("anim_mode")``) and branch on that, so a naive regex
    over ``anim_mode == '...'`` misses them. We parse the AST and collect every
    string literal that is compared against (== / != / ``in``) the anim-mode
    variable — however it is named — then the ``default`` from the param spec.

    Returns the ordered mode list (default first) or ``None`` if the method has
    no anim_mode param or we couldn't find ≥2 modes.
    """
    if fn is None or not hasattr(fn, "__code__"):
        return None
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None

    # 1) Find the local name bound to anim_mode (direct, or via params.get).
    anim_var = "anim_mode"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            val = node.value
            if isinstance(tgt, ast.Name) and isinstance(val, ast.Call):
                # mode = params.get("anim_mode", "none")
                if isinstance(val.func, ast.Attribute) and val.func.attr == "get":
                    args = val.args
                    if args and isinstance(args[0], ast.Constant) and args[0].value == "anim_mode":
                        anim_var = tgt.id
            elif isinstance(tgt, ast.Name) and isinstance(val, ast.Name) and val.id == "anim_mode":
                anim_var = tgt.id

    found: list[str] = []
    seen: set[str] = set()

    def add(v):
        if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_\-]+", v) and v not in seen:
            seen.add(v)
            found.append(v)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # Identify the operand that is the anim-mode variable
            operands = [node.left] + list(node.comparators)
            names = {o.id for o in operands if isinstance(o, ast.Name)}
            if anim_var in names:
                for o in operands:
                    if isinstance(o, ast.Constant) and isinstance(o.value, str):
                        add(o.value)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            # also catch `if mode in ("a", "b")`
            operands = [node.test.left] + list(node.test.comparators)
            names = {o.id for o in operands if isinstance(o, ast.Name)}
            if anim_var in names:
                for o in operands:
                    if isinstance(o, ast.Constant) and isinstance(o.value, str):
                        add(o.value)
                    elif isinstance(o, (ast.Tuple, ast.List, ast.Set)):
                        for elt in o.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                add(elt.value)
    return found if len(found) >= 2 else None


def _enrich_params(params: dict | None, fn=None) -> dict | None:
    """Inject 'choices' into param specs where the description encodes an enum
    list, or (for ``anim_mode``) where the method's own source declares them.

    Also normalises the legacy ``options`` key (some older methods used it
    instead of ``choices``) to ``choices`` so the UI select factory picks it up.
    """
    if not params:
        return params
    result = {}
    for key, spec in params.items():
        if not isinstance(spec, dict):
            result[key] = spec
            continue
        # Normalise legacy 'options' -> 'choices' (UI reads 'choices').
        if 'choices' not in spec and 'options' in spec and isinstance(spec['options'], (list, tuple)) and spec['options']:
            spec = {**spec, 'choices': list(spec['options'])}
        if 'choices' in spec:
            # Already explicit — keep as-is (data wins).
            result[key] = spec
            continue
        choices = _parse_choices(spec.get('description', ''), spec.get('default'))
        if choices is None and key == "anim_mode" and fn is not None:
            # Recover the mode enum from the method body itself.
            derived = _derive_anim_mode_choices(fn)
            default = spec.get("default")
            if derived and default and default in derived:
                # Ensure default leads the list (UI convention).
                ordered = [default] + [m for m in derived if m != default]
                choices = ordered
            else:
                choices = derived
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
            "params": _enrich_params(meta.params, meta.fn),
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
        """Block until a frame newer than last_id exists, or timeout."""
        with _LIVE_FRAME_COND:
            if _LIVE_FRAME is not None and _LIVE_FRAME_ID != last_id:
                return
            _LIVE_FRAME_COND.wait(timeout=1.0)

    async def generate():
        # None = nothing delivered yet. Must not be a sentinel int: _LIVE_FRAME_ID
        # starts at 0, so a -1 sentinel reads as "new frame available" before the
        # live loop has published anything.
        last_id = None
        loop = asyncio.get_event_loop()
        while True:
            with _LIVE_FRAME_LOCK:
                fid = _LIVE_FRAME_ID
                data = _LIVE_FRAME
            if data is None or fid == last_id:
                # Nothing new. This is the only path back to the top of the
                # loop, and it always awaits — otherwise the generator spins
                # and starves the event loop, wedging the whole server.
                await loop.run_in_executor(None, _wait_for_new_frame, last_id)
                continue
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
    # `get_all_node_defs` is @cache'd and hands back the SHARED dict. Enriching
    # it in place permanently rewrote core's node-def contract for every other
    # consumer (executor port derivation, tests) and made that state depend on
    # whether this endpoint had been hit yet. Choice-enrichment is presentation
    # only — build a shallow per-node copy so it never reaches the model.
    #
    # Map method_id -> function so enrichment can derive anim_mode choices
    # from each method's own source (handles aliased anim-mode variables).
    _meta_by_id = registry.get_all()
    out: dict[str, dict] = {}
    for _key, nd in defs.items():
        if nd.get('params'):
            _meta = _meta_by_id.get(nd.get('method_id', ''))
            out[_key] = {**nd,
                         'params': _enrich_params(nd['params'],
                                                  _meta.fn if _meta else None)}
        else:
            out[_key] = nd
    return out


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
    fps: float = 24.0        # timeline FPS — the live cook-rate limiter's target
    fps_limit: bool = False  # when True the live loop paces its cooks to `fps`


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
    # cook-rate limiter
    "fps_limit": False, "target_fps": 0.0,
}

# Live cook-rate limiter. Deliberately NOT captured by the loop closure: the
# loop re-reads it every frame, so retuning the rate is a hot-swap (a running
# render keeps its executor and sim caches) instead of a thread restart.
LIVE_DISPLAY_FPS_CAP = 30.0
_live_rate: dict = {"limit": False, "fps": 24.0}

# Persistent executor — survives hot-swaps so Arch-A sim caches are kept
_live_executor = None          # GraphExecutor | None
_live_last_nodes: list = []
_live_last_edges: list = []
_live_last_gid:   str  = ""
_live_last_seed:  int  = 0
_live_last_canvas: tuple = (0, 0)

# Serializes every use of the shared _live_executor. The loop holds it for the
# duration of a frame's cook; hot-swap cache surgery takes it too, so neither an
# orphaned loop nor a concurrent POST can ever cook on the executor at the same
# time as the current loop (GraphExecutor is not thread-safe).
_live_exec_lock = threading.RLock()
# Bumped on every start/stop. A loop whose captured generation is stale exits at
# the top of its next iteration and never pushes another frame — so an orphan
# that outlived its join() cannot interleave its frames into the preview.
_live_gen = 0

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

    frames == 0 stops; anything else starts or hot-swaps.

    Only one live loop ever renders. The loop re-reads the shared graph doc
    every frame, so an edited graph is absorbed by the *running* loop — a
    param tweak must never spawn a second render. A thread restart happens
    only when the request changes something the loop captured at start time
    (graph id, canvas, seed) or when no loop is running.

    The GraphExecutor is kept alive across hot-swaps. Arch-A simulation caches
    survive unchanged hot-swaps; only nodes with changed non-volatile params
    are invalidated.
    """
    global _live_sim_cancel, _live_sim_thread, _live_gen
    global _live_executor, _live_last_nodes, _live_last_edges, _live_last_gid, _live_last_seed, _live_last_canvas
    with _live_sim_lock:
        running = _live_sim_thread is not None and _live_sim_thread.is_alive()

        if req.frames == 0:
            # Retire the generation first: a loop stuck mid-cook past the join
            # timeout still exits on its next iteration and pushes nothing.
            _live_gen += 1
            if running:
                _live_sim_cancel.set()
                _live_sim_thread.join(timeout=5.0)
            _live_sim_thread = None
            return {"status": "stopped"}

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

        # ── Cook-rate limiter ─────────────────────────────────────────
        # Off: cook as fast as the graph allows, capped at the display rate.
        # On: cook one frame per 1/fps second so a heavy graph stops burning
        # the machine on frames nobody sees, and live runs at the tempo it
        # will export at. Written here (not into the loop closure) so this is
        # a hot-swap — the running loop picks it up on its next frame.
        _live_rate["limit"] = bool(req.fps_limit)
        _live_rate["fps"] = max(0.1, float(req.fps or 24.0))
        _live_stats["fps_limit"]  = _live_rate["limit"]
        _live_stats["target_fps"] = _live_rate["fps"] if _live_rate["limit"] else 0.0

        # ── Hot-swap vs restart ───────────────────────────────────────
        # gid, canvas and seed are captured by the loop closure (and decide
        # executor identity), so changing one needs a new thread. Everything
        # else — params, wiring, node adds/removes — is re-read from the doc
        # by the running loop, so the POST must not touch the thread at all.
        # Restarting there was the bug: a slow frame outlived the 5s join and
        # every further param edit stacked another live loop on the same
        # executor, until the renders interleaved in the preview and wedged.
        hot_swap = (
            running
            and _live_executor is not None
            and gid == _live_last_gid
            and (width, height) == _live_last_canvas
            and seed == _live_last_seed
        )

        if not hot_swap:
            # Retire the old loop before touching the executor, so a full
            # flush can't land between two of its frames.
            _live_gen += 1
            if running:
                _live_sim_cancel.set()
                _live_sim_thread.join(timeout=5.0)
                if _live_sim_thread.is_alive():
                    # Stuck mid-cook. Safe to proceed: the stale generation
                    # stops it from pushing another frame, and _live_exec_lock
                    # keeps the new loop from cooking until it lets go.
                    print("[live-sim] previous loop still finishing its frame "
                          "— retired by generation, not restarted")
            _live_sim_thread = None

        # ── Persistent executor: reuse or create ──────────────────────
        # Held across the cache surgery so invalidation can't race a cook.
        with _live_exec_lock:
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

        if hot_swap:
            # The running loop picks the new doc up on its next frame.
            print(f"[live-sim] hot-swap into running loop, {_inv_msg}")
            return {"status": "running", "hot_swap": True}

        # ── Restart: the old loop is already retired above ────────────
        cancel = threading.Event()
        my_gen = _live_gen

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
            print(f"[live-sim] starting loop gen={my_gen}, {len(nodes)} nodes, "
                  f"{len(edges)} edges, {_inv_msg}")
            while not cancel.is_set():
                # A newer loop (or a stop) has taken over — leave without
                # cooking or pushing, so two loops never render at once.
                if _live_gen != my_gen:
                    print(f"[live-sim] gen {my_gen} superseded — exiting")
                    break
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

                    # The executor is shared with hot-swap invalidation and
                    # with any loop still winding down — one cook at a time.
                    with _live_exec_lock:
                        flat_outputs, terminal_id, node_errors = executor.execute(
                            work_nodes, work_edges, seed, frame=frame % LIVE_TOTAL_FRAMES, frames=LIVE_TOTAL_FRAMES
                        )
                    # Blocking on that lock can take a whole frame; re-check
                    # before publishing anything.
                    if _live_gen != my_gen or cancel.is_set():
                        break
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
                # Pace the next cook: the timeline FPS when the limiter is on,
                # otherwise ~30fps so the browser can display each frame. Read
                # per frame, so a limiter/FPS change retunes this loop in place.
                _target_fps = (_live_rate["fps"] if _live_rate["limit"]
                               else LIVE_DISPLAY_FPS_CAP)
                _frame_interval = 1.0 / max(0.1, _target_fps)
                _elapsed = time.monotonic() - _tick_start
                _sleep = _frame_interval - _elapsed
                if _sleep > 0:
                    # Wake early on cancel so a stop is not stuck behind a long
                    # limiter interval (1 fps ⇒ a whole second of dead sleep).
                    cancel.wait(_sleep)

        _live_sim_cancel = cancel
        # Named so a stacked loop is visible in threading.enumerate() / a
        # stack dump instead of hiding behind "Thread-37".
        _live_sim_thread = threading.Thread(
            target=_live_loop, name=f"live-sim-{my_gen}", daemon=True)
        _live_sim_thread.start()
        return {"status": "running"}


@app.get("/api/graph/live/status")
def live_graph_status():
    running = _live_sim_thread is not None and _live_sim_thread.is_alive()
    # `loops` is the stacked-render canary: it must be 0 or 1. Anything higher
    # means a retired loop is still winding down (or, if it stays high, that
    # something restarted the loop instead of hot-swapping it).
    loops = sum(1 for t in threading.enumerate()
                if t.name.startswith("live-sim") and t.is_alive())
    return {"running": running, "loops": loops, "gen": _live_gen, **_live_stats}


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


class SeqFrameUpload(BaseModel):
    frame: int
    data: str            # base64 PNG, with or without a data: URL prefix
    reset: bool = False  # first frame of a run — clear the directory first


@app.post("/api/sequences/{name}/frames")
def put_sequence_frame(name: str, req: SeqFrameUpload):
    """Store one browser-rendered frame in a sequence directory.

    Client-rendered graphs (3D / p5) cook on the browser GPU, so the server
    never sees their pixels. Without somewhere to put them a client run can
    only ever show one live canvas — there is no sequence for the timeline to
    scrub, and so no clip. This is the client engine's equivalent of the
    per-frame PNG write in the server run job.

    `reset` wipes the directory first so a shorter re-render cannot leave the
    tail of a previous longer run trailing off the end of the new clip.
    """
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    seq_dir = SEQUENCES_DIR / name
    if req.reset and seq_dir.exists():
        shutil.rmtree(str(seq_dir))
    seq_dir.mkdir(parents=True, exist_ok=True)

    try:
        blob = base64.b64decode(req.data.split(",", 1)[-1])
    except Exception as exc:
        raise HTTPException(400, f"Bad base64 frame data: {exc}")

    png_path = seq_dir / f"frame_{req.frame:04d}.png"
    png_path.write_bytes(blob)
    # get_sequence_frame serves a cached JPEG sibling whenever one exists; a
    # stale one from a previous run would shadow the frame just written.
    jpg_path = png_path.with_suffix(".jpg")
    if jpg_path.exists():
        jpg_path.unlink()
    return {"ok": True, "frame": req.frame}


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
def _hermes_venv_python(agent_dir: Path) -> Path | None:
    """The venv interpreter inside a hermes-agent checkout, either layout.

    Windows venvs put the interpreter in ``venv/Scripts/python.exe``; POSIX
    uses ``venv/bin/python``. Probing both means the same code works on either
    platform instead of silently reporting "backend not found" on Windows.
    """
    for rel in (("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
        cand = agent_dir.joinpath(*rel)
        if cand.exists():
            return cand
    return None


def _resolve_hermes() -> tuple[Path, Path | None]:
    """Locate the hermes-agent checkout and its interpreter.

    CONFIGURATION.md advertises HERMES_PYTHON as "Auto-detected", which was
    aspirational: the default was a hardcoded POSIX path under a single
    hardcoded directory, so any install that was not at ~/.hermes/hermes-agent
    with a POSIX venv reported "backend not found" and Node Doctor was dead.

    Resolution order — explicit config always wins:
      1. HERMES_AGENT_DIR, if set.
      2. Known install locations, in order of how standard they are.
    The interpreter is HERMES_PYTHON if set, else the venv inside whichever
    directory we settled on.
    """
    explicit = os.environ.get("HERMES_AGENT_DIR")
    if explicit:
        agent_dir = Path(explicit)
    else:
        local_appdata = os.environ.get("LOCALAPPDATA")
        candidates = [
            Path.home() / ".hermes" / "hermes-agent",
            Path.home() / "AppData" / "Local" / "hermes" / "hermes-agent",
            Path.home() / "hermes" / "hermes-agent",
        ]
        if local_appdata:
            candidates.insert(1, Path(local_appdata) / "hermes" / "hermes-agent")
        # Prefer a candidate that actually has an interpreter; fall back to the
        # documented default so the error message still names the expected path.
        agent_dir = next(
            (c for c in candidates if _hermes_venv_python(c) is not None),
            candidates[0],
        )

    env_py = os.environ.get("HERMES_PYTHON")
    py = Path(env_py) if env_py else _hermes_venv_python(agent_dir)
    return agent_dir, py


HERMES_AGENT_DIR, _HERMES_PY_OPT = _resolve_hermes()
# Keep the documented default in the message when nothing was found, so the
# warning tells the user which path to point HERMES_AGENT_DIR at.
_HERMES_PY = _HERMES_PY_OPT or (HERMES_AGENT_DIR / "venv" / "bin" / "python")

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
        # nd_runner.py resolves HERMES_AGENT_DIR independently and defaults to
        # ~/.hermes/hermes-agent. When the server AUTO-detected a different
        # install, that default is wrong, so pass the resolved directory down
        # explicitly — otherwise the child imports from a path that isn't there.
        _child_env = {**os.environ, "HERMES_AGENT_DIR": str(HERMES_AGENT_DIR)}
        proc = await asyncio.create_subprocess_exec(
            str(_HERMES_PY), str(_ND_RUNNER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env,
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
    args = parser.parse_args()

    uvicorn.run("image_pipeline.server:app", host="0.0.0.0", port=args.port, reload=False, log_config=None)
