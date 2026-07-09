"""Profiling script: measure per-frame cost breakdown for a representative live graph.

Graph: Procedural Noise (#05) → Glitch Art (#17, uses load_input) → Transform (uses _input_image) → output

Buckets measured:
  - node_exec_ms:    actual meta.fn() call time per node, summed
  - disk_io_ms:      _input.png write time (the legacy passthrough)
  - encode_ms:       JPEG encode + _push_live_frame equivalent (cap + encode)
  - overhead_ms:     everything else (topological sort, dirty checks, readback, etc.)

Run with:
    cd /Users/admin/Documents/GitHub/grillmaster-command-center
    python -m image_pipeline.tests.profile_live
"""
from __future__ import annotations
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import get_meta
from image_pipeline.core.utils import set_canvas, W, H

# ---------------------------------------------------------------------------
# Patching hooks — we monkey-patch at the PIL.Image.save level so we can
# time exactly the _input.png write that happens in graph.py:774 without
# modifying graph.py itself for the profiling run.
# ---------------------------------------------------------------------------

_io_time_acc: list[float] = []

_original_pil_save = Image.Image.save

def _timed_pil_save(self, fp, *args, **kwargs):
    name = str(fp) if isinstance(fp, (str, Path)) else ""
    if "_input.png" in name:
        t0 = time.perf_counter()
        _original_pil_save(self, fp, *args, **kwargs)
        _io_time_acc.append(time.perf_counter() - t0)
    else:
        _original_pil_save(self, fp, *args, **kwargs)

Image.Image.save = _timed_pil_save

_node_exec_time_acc: dict[str, list[float]] = {}
_node_readback_time_acc: dict[str, list[float]] = {}

# ---------------------------------------------------------------------------
# Graph definition: Noise → Glitch (load_input) → Transform (_input_image)
# ---------------------------------------------------------------------------

NODES = [
    {
        "id": "src",
        "method_id": "05",   # Procedural Noise
        "params": {"noise_type": "perlin", "scale": 4.0},
        "dirty": True,
        "render": False,
    },
    {
        "id": "glitch",
        "method_id": "17",   # Glitch Art (uses load_input)
        "params": {"glitch_type": "classic", "intensity": 0.5},
        "dirty": True,
        "render": False,
    },
    {
        "id": "xform",
        "method_id": "__transform__",  # Transform (uses _input_image)
        "params": {"rotate": 5.0, "scale": 0.95},
        "dirty": True,
        "render": True,
    },
]

EDGES = [
    {"src_node": "src",   "src_port": "image", "dst_node": "glitch", "dst_port": "image_in"},
    {"src_node": "glitch","src_port": "image", "dst_node": "xform",  "dst_port": "image_in"},
]

LIVE_TOTAL_FRAMES = 300
WARMUP_RUNS = 3
BENCH_RUNS = 20
WIDTH, HEIGHT = 768, 512


def _encode_frame_ms(arr: np.ndarray) -> float:
    """Time one JPEG encode at live-push quality (same params as _push_live_frame)."""
    from io import BytesIO
    import cv2
    t0 = time.perf_counter()
    h, w = arr.shape[:2]
    if w > 1280:
        scale = 1280 / w
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        display = arr
    u8 = (np.clip(display, 0, 1) * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(u8).save(buf, format="JPEG", quality=85)
    return (time.perf_counter() - t0) * 1000.0


def run_bench(out: Path) -> dict[str, Any]:
    set_canvas(WIDTH, HEIGHT)
    # Same config as the server's live loop: memory transport, no audit writes.
    ex = GraphExecutor(out, in_memory=True, audit_to_disk=False)

    # Patch the registry to time individual node fn calls
    from image_pipeline.core.registry import get_meta as _get_meta
    _node_times: dict[str, list[float]] = {"src": [], "glitch": [], "xform": []}
    method_to_node = {"05": "src", "17": "glitch", "__transform__": "xform"}

    # We'll time inside execute by patching meta.fn for each method
    _orig_fns: dict[str, Any] = {}

    def _make_timed(method_id: str, orig_fn):
        label = method_to_node.get(method_id, method_id)
        def _wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = orig_fn(*args, **kwargs)
            _node_times[label].append((time.perf_counter() - t0) * 1000.0)
            return result
        return _wrapper

    for mid in ("05", "17", "__transform__"):
        m = _get_meta(mid)
        _orig_fns[mid] = m.fn
        object.__setattr__(m, "fn", _make_timed(mid, m.fn))

    results = {
        "total_ms": [],
        "encode_ms": [],
        "disk_io_ms": [],
        "node_src_ms": [],
        "node_glitch_ms": [],
        "node_xform_ms": [],
    }

    try:
        for run_i in range(WARMUP_RUNS + BENCH_RUNS):
            _io_time_acc.clear()
            for v in _node_times.values():
                v.clear()

            nodes = [dict(n) for n in NODES]
            for n in nodes:
                n["dirty"] = True
                n.setdefault("params", {})["time"] = float(run_i)

            t_frame_start = time.perf_counter()
            flat, _tl, errs = ex.execute(
                nodes, EDGES, seed=42,
                frame=run_i % LIVE_TOTAL_FRAMES,
                frames=LIVE_TOTAL_FRAMES,
            )
            t_frame_end = time.perf_counter()

            if errs:
                print(f"[WARN] run {run_i} errors: {errs}")
                continue

            terminal_img = flat.get("xform", {}).get("image")
            enc_ms = _encode_frame_ms(terminal_img) if terminal_img is not None else 0.0
            total_ms = (t_frame_end - t_frame_start) * 1000.0
            io_ms = sum(_io_time_acc) * 1000.0

            if run_i >= WARMUP_RUNS:
                results["total_ms"].append(total_ms)
                results["encode_ms"].append(enc_ms)
                results["disk_io_ms"].append(io_ms)
                results["node_src_ms"].append(sum(_node_times["src"]))
                results["node_glitch_ms"].append(sum(_node_times["glitch"]))
                results["node_xform_ms"].append(sum(_node_times["xform"]))
    finally:
        for mid, orig in _orig_fns.items():
            object.__setattr__(_get_meta(mid), "fn", orig)

    return results


def main():
    out = Path(tempfile.mkdtemp(prefix="gm_prof_"))
    try:
        print(f"Profiling {BENCH_RUNS} frames at {WIDTH}×{HEIGHT}  (graph: Noise→Glitch→Transform)")
        print(f"Warmup: {WARMUP_RUNS} frames\n")

        r = run_bench(out)

        total    = np.array(r["total_ms"])
        encode   = np.array(r["encode_ms"])
        disk_io  = np.array(r["disk_io_ms"])

        # node_exec = total - disk_io - encode - overhead_etc
        # We can't perfectly isolate node exec without patching the fn call,
        # but disk_io and encode are directly measured. "other" = everything else
        # including the node fn call, readback, sort, etc.
        other    = total - disk_io - encode

        def fmt(arr):
            return f"{arr.mean():.1f}ms  ±{arr.std():.1f}ms  (min {arr.min():.1f} max {arr.max():.1f})"

        src_ms    = np.array(r["node_src_ms"])
        glitch_ms = np.array(r["node_glitch_ms"])
        xform_ms  = np.array(r["node_xform_ms"])

        node_total = src_ms + glitch_ms + xform_ms
        executor_overhead = total - disk_io - encode - node_total

        print(f"Frame total                             {fmt(total)}")
        print(f"  Node: Procedural Noise  (#05)         {fmt(src_ms)}   {src_ms.mean()/total.mean()*100:.0f}%")
        print(f"  Node: Glitch Art        (#17)         {fmt(glitch_ms)}   {glitch_ms.mean()/total.mean()*100:.0f}%")
        print(f"  Node: Transform         (#xform)      {fmt(xform_ms)}   {xform_ms.mean()/total.mean()*100:.0f}%")
        print(f"  Disk I/O (_input.png write ×2)        {fmt(disk_io)}   {disk_io.mean()/total.mean()*100:.0f}%")
        print(f"  JPEG encode + push                    {fmt(encode)}   {encode.mean()/total.mean()*100:.0f}%")
        print(f"  Executor overhead (sort/readback/etc) {fmt(executor_overhead)}   {executor_overhead.mean()/total.mean()*100:.0f}%")
        print(f"\nApprox live FPS (uncapped):  {1000/total.mean():.1f}  (target: 30)")

    finally:
        shutil.rmtree(out, ignore_errors=True)
        Image.Image.save = _original_pil_save


if __name__ == "__main__":
    main()
