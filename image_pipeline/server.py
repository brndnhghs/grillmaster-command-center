"""FastAPI server for the image-generation pipeline GUI."""
from __future__ import annotations
import base64
import io
import json
import queue
import sys
import threading
import time
import uuid
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
from image_pipeline.core.animation import JobCancelled
from image_pipeline.core.graph import GraphExecutor, GraphError
import image_pipeline.methods  # noqa: F401

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(title="Image Pipeline")
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
            "params": meta.params,
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


@app.get("/api/node-defs")
def get_node_defs():
    from image_pipeline.core.graph import get_all_node_defs
    return get_all_node_defs()


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
        executor = GraphExecutor(out_dir)
        terminal_frames: list = []
        terminal_node_id: str | None = None

        n_frames = max(1, frames)
        for frame in range(n_frames):
            if cancel_event.is_set():
                q.put(("error", "Cancelled"))
                return

            print(f"  Frame {frame + 1}/{n_frames}")
            try:
                flat_outputs, terminal_id = executor.execute(
                    nodes, edges, seed, frame=frame, frames=n_frames
                )
            except GraphError as exc:
                q.put(("error", str(exc)))
                return

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
                q.put(("done", {"rel_path": rel_path, "type": "video", "frames": len(terminal_frames)}))
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
            q.put(("done", {"rel_path": rel_path, "type": "image", "frames": 1}))

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

    uvicorn.run("image_pipeline.server:app", host="0.0.0.0", port=args.port, reload=False)
