"""FastAPI server for the image-generation pipeline GUI."""
from __future__ import annotations
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
import image_pipeline.methods  # noqa: F401

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

app = FastAPI(title="Image Pipeline")
app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")

# ── In-memory job store ───────────────────────────────────────────────

_jobs: dict[str, dict] = {}


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


def _run_job(job_id, method_id, seed, params, animate, fps, duration, out_dir):
    job = _jobs[job_id]
    q = job["q"]
    messages: list[str] = []

    meta = registry.get_meta(method_id)
    if meta is None:
        q.put(("error", f"Method '{method_id}' not found"))
        q.put(None)
        return

    writer = _QueueWriter(q, messages)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = writer  # type: ignore[assignment]
    sys.stderr = writer  # type: ignore[assignment]

    try:
        if animate:
            from image_pipeline.core.animation import animate_method
            out_path = animate_method(
                meta, out_dir, seed,
                fps=fps, duration=duration,
                user_params=params,
            )
            if out_path and Path(out_path).exists():
                job["output_path"] = str(out_path)
                job["type"] = "video"
                q.put(("done", {"output_path": str(out_path), "type": "video"}))
            else:
                q.put(("error", "Animation failed — method has no natural animation frames"))
        else:
            try:
                meta.fn(out_dir, seed, params=params)
            except TypeError:
                meta.fn(out_dir, seed)

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
        sys.stdout = old_out
        sys.stderr = old_err
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


# ── Entry point ───────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_pipeline.server:app", host="0.0.0.0", port=7860, reload=False)
