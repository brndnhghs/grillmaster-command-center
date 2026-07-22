"""Profiling script: measure per-frame cost breakdown for a representative live graph.

Graph: Procedural Noise (#05) → Glitch Art (#17, uses load_input) → Transform (uses _input_image)

TWO MODES — the distinction is the whole point of this script:

  live  (default)  Reproduces ``_live_loop``'s Phase-6 selective dirty marking:
                   only time-varying nodes, Architecture-A sims, nodes whose
                   user params changed, and nodes with paramKeyframes are
                   dirtied, then the set is cascaded forward with
                   ``_compute_live_dirty``. This is what the live preview
                   actually does, so this is the number that describes the
                   product.

  cold             Forces ``dirty=True`` on every node every frame. This is
                   pre-Phase-6 invariant 1, which DESIGN.md explicitly relaxed.
                   It measures a worst-case cold frame — useful as an upper
                   bound and for scrub/seek behaviour, but it is NOT live mode.

Until 2026-07-22 this script only did the `cold` thing while presenting the
result as "Approx live FPS". That understated live performance by whatever the
incremental-recook skip saves, so every optimisation decision taken against
this number was steering by an architecture the executor had already moved
away from. If you touch this file, keep the two modes honest and distinctly
labelled.

Buckets measured:
  - node_exec_ms:    actual meta.fn() call time per node, summed (0 when skipped)
  - disk_io_ms:      legacy per-edge transport write (see _TRANSPORT_NAMES)
  - encode_ms:       JPEG encode + _push_live_frame equivalent (cap + encode)
  - overhead_ms:     everything else (topological sort, dirty checks, readback)

Run with:
    python -m image_pipeline.tests.profile_live            # both modes
    python -m image_pipeline.tests.profile_live live       # live only
    python -m image_pipeline.tests.profile_live cold       # cold only
"""
from __future__ import annotations
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.arch import detect_architecture
from image_pipeline.core.graph import GraphExecutor, _compute_live_dirty
from image_pipeline.core.registry import get_meta
from image_pipeline.core.utils import set_canvas, W, H

# ---------------------------------------------------------------------------
# Patching hooks — we monkey-patch at the PIL.Image.save level so we can time
# exactly the legacy per-edge transport write that happens in graph.py without
# modifying graph.py itself for the profiling run.
# ---------------------------------------------------------------------------

# The executor has written this file as .png historically and .bmp since the
# transport-cost fix; match both so the profiler keeps working either way.
_TRANSPORT_NAMES = ("_input.png", "_input.bmp")

_io_time_acc: list[float] = []

_original_pil_save = Image.Image.save


def _timed_pil_save(self, fp, *args, **kwargs):
    name = str(fp) if isinstance(fp, (str, Path)) else ""
    if any(t in name for t in _TRANSPORT_NAMES):
        t0 = time.perf_counter()
        _original_pil_save(self, fp, *args, **kwargs)
        _io_time_acc.append(time.perf_counter() - t0)
    else:
        _original_pil_save(self, fp, *args, **kwargs)


Image.Image.save = _timed_pil_save

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

# Volatile keys excluded from the per-node param-change check, mirroring the
# live loop (and _node_params_hash in graph.py).
_VOLATILE = ("time", "frame", "frame_seed", "_timeline", "_input_image", "input_image")


def _mark_dirty_live(nodes, edges, frame: int, last_params: dict[str, dict]) -> set[str]:
    """Reproduce _live_loop's Phase-6 selective dirty marking, verbatim.

    Kept deliberately close to the server implementation — if that logic
    changes and this does not, the profiler silently stops describing the
    product again.
    """
    initially_dirty: set[str] = set()
    for n in nodes:
        nid = n["id"]
        n.setdefault("params", {})

        meta = get_meta(n.get("method_id", ""))
        is_tv = True if meta is None else meta.is_time_varying
        # Architecture-A sims always re-cook: their sim-cache frame index advances.
        if meta is not None and detect_architecture(meta) == "A":
            is_tv = True

        if is_tv:
            # Inject time ONLY into time-varying nodes, so static nodes' param
            # hash stays stable across frames (that stability is what lets the
            # executor skip them).
            n["params"]["time"] = float(frame)
            initially_dirty.add(nid)
        else:
            cur = {k: v for k, v in n["params"].items() if k not in _VOLATILE}
            if cur != last_params.get(nid):
                initially_dirty.add(nid)
            last_params[nid] = dict(cur)

        if n.get("paramKeyframes"):
            initially_dirty.add(nid)

    return _compute_live_dirty(nodes, edges, initially_dirty)


def _mark_dirty_cold(nodes, edges, frame: int, last_params: dict[str, dict]) -> set[str]:
    """Pre-Phase-6 behaviour: everything re-cooks every frame."""
    for n in nodes:
        n.setdefault("params", {})["time"] = float(frame)
    return {n["id"] for n in nodes}


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


def run_bench(out: Path, mode: str = "live") -> dict[str, Any]:
    set_canvas(WIDTH, HEIGHT)
    # Same config as the server's live loop: memory transport, no audit writes,
    # and ONE persistent executor across frames (the in-memory skip depends on
    # _prev_outputs surviving between frames).
    ex = GraphExecutor(out, in_memory=True, audit_to_disk=False)
    mark = _mark_dirty_live if mode == "live" else _mark_dirty_cold

    _node_times: dict[str, list[float]] = {"src": [], "glitch": [], "xform": []}
    method_to_node = {"05": "src", "17": "glitch", "__transform__": "xform"}
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
        m = get_meta(mid)
        _orig_fns[mid] = m.fn
        object.__setattr__(m, "fn", _make_timed(mid, m.fn))

    results: dict[str, list] = {
        "total_ms": [], "encode_ms": [], "disk_io_ms": [],
        "node_src_ms": [], "node_glitch_ms": [], "node_xform_ms": [],
        "cooked": [], "skipped": [],
    }

    # Persistent across frames, exactly like the live loop's _last_params.
    last_params: dict[str, dict] = {}
    nodes = [dict(n) for n in NODES]
    for n in nodes:
        n["params"] = dict(n.get("params") or {})

    try:
        for run_i in range(WARMUP_RUNS + BENCH_RUNS):
            _io_time_acc.clear()
            for v in _node_times.values():
                v.clear()

            frame = run_i % LIVE_TOTAL_FRAMES
            dirty_set = mark(nodes, EDGES, frame, last_params)
            for n in nodes:
                n["dirty"] = n["id"] in dirty_set

            t_frame_start = time.perf_counter()
            flat, _tl, errs = ex.execute(
                nodes, EDGES, seed=42, frame=frame, frames=LIVE_TOTAL_FRAMES,
            )
            t_frame_end = time.perf_counter()

            if errs:
                print(f"[WARN] run {run_i} errors: {errs}")
                continue

            terminal_img = flat.get("xform", {}).get("image")
            enc_ms = _encode_frame_ms(terminal_img) if terminal_img is not None else 0.0
            total_ms = (t_frame_end - t_frame_start) * 1000.0
            io_ms = sum(_io_time_acc) * 1000.0
            stats = getattr(ex, "last_frame_stats", {}) or {}

            if run_i >= WARMUP_RUNS:
                results["total_ms"].append(total_ms)
                results["encode_ms"].append(enc_ms)
                results["disk_io_ms"].append(io_ms)
                results["node_src_ms"].append(sum(_node_times["src"]))
                results["node_glitch_ms"].append(sum(_node_times["glitch"]))
                results["node_xform_ms"].append(sum(_node_times["xform"]))
                results["cooked"].append(stats.get("nodes_cooked", 0))
                results["skipped"].append(stats.get("nodes_skipped", 0))
    finally:
        for mid, orig in _orig_fns.items():
            object.__setattr__(get_meta(mid), "fn", orig)

    return results


def _report(mode: str, r: dict[str, Any]) -> float:
    total = np.array(r["total_ms"])
    encode = np.array(r["encode_ms"])
    disk_io = np.array(r["disk_io_ms"])
    src_ms = np.array(r["node_src_ms"])
    glitch_ms = np.array(r["node_glitch_ms"])
    xform_ms = np.array(r["node_xform_ms"])
    node_total = src_ms + glitch_ms + xform_ms
    overhead = total - disk_io - encode - node_total

    def fmt(a):
        return f"{a.mean():7.1f}ms  ±{a.std():5.1f}  (min {a.min():.1f} max {a.max():.1f})"

    def pct(a):
        return f"{a.mean() / total.mean() * 100:3.0f}%"

    label = ("LIVE  — Phase-6 selective recook (what the preview does)"
             if mode == "live" else
             "COLD  — every node forced dirty (worst case / scrub)")
    print(f"\n=== {label} ===")
    print(f"  Frame total                            {fmt(total)}")
    print(f"    Node: Procedural Noise  (#05)        {fmt(src_ms)}  {pct(src_ms)}")
    print(f"    Node: Glitch Art        (#17)        {fmt(glitch_ms)}  {pct(glitch_ms)}")
    print(f"    Node: Transform         (#xform)     {fmt(xform_ms)}  {pct(xform_ms)}")
    print(f"    Legacy edge transport write          {fmt(disk_io)}  {pct(disk_io)}")
    print(f"    JPEG encode + push                   {fmt(encode)}  {pct(encode)}")
    print(f"    Executor overhead (sort/readback)    {fmt(overhead)}  {pct(overhead)}")
    print(f"  nodes cooked/frame {np.mean(r['cooked']):.1f}   "
          f"skipped/frame {np.mean(r['skipped']):.1f}   (of {len(NODES)})")
    fps = 1000 / total.mean()
    print(f"  Approx FPS (uncapped): {fps:.1f}   (target 30)")
    return fps


def main():
    modes = [a for a in sys.argv[1:] if a in ("live", "cold")] or ["live", "cold"]
    print(f"Profiling {BENCH_RUNS} frames at {WIDTH}x{HEIGHT}  "
          f"(graph: Noise -> Glitch -> Transform)")
    print(f"Warmup: {WARMUP_RUNS} frames")

    fps: dict[str, float] = {}
    for mode in modes:
        out = Path(tempfile.mkdtemp(prefix=f"gm_prof_{mode}_"))
        try:
            fps[mode] = _report(mode, run_bench(out, mode))
        finally:
            shutil.rmtree(out, ignore_errors=True)

    if len(fps) == 2:
        print(f"\nIncremental recook is worth {fps['live'] / fps['cold']:.1f}x on this graph "
              f"({fps['cold']:.1f} -> {fps['live']:.1f} fps).")
        print("Optimise against the LIVE row — COLD is an upper bound, not the product.")


if __name__ == "__main__":
    try:
        main()
    finally:
        Image.Image.save = _original_pil_save
