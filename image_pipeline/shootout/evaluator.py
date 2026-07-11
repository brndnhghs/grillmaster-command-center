"""Render a genome to a short mp4 and reject dead clips (plan §7).

Rendering happens in-process (same internals as /api/graph/render-sequence:
GraphExecutor per frame), but pipes frames straight into ffmpeg via
frames_to_mp4 instead of writing per-frame PNGs — the mp4 is the durable
artifact (determinism-epoch policy, plan §12). Liveness stats are computed
on a spatially-downsampled grayscale copy of every frame, so memory stays
flat regardless of clip size.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from image_pipeline.core.animation import frames_to_mp4
from image_pipeline.core.graph import GraphExecutor

from .config import ShootoutConfig, DEFAULT_CONFIG

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"
SEQUENCES_DIR = OUTPUT_ROOT / "sequences"


def seq_name_for(genome_id: str) -> str:
    return f"shootout-{genome_id}"


# ── Liveness stats ────────────────────────────────────────────────────


class LivenessAccumulator:
    """Streaming stats over downsampled grayscale frames."""

    def __init__(self, cfg: ShootoutConfig):
        self.cfg = cfg
        self.small: list[np.ndarray] = []
        self.nan = False
        self.missing = 0
        self.total = 0

    def add(self, arr: np.ndarray | None) -> None:
        self.total += 1
        if arr is None:
            self.missing += 1
            return
        s = self.cfg.stat_stride
        small = np.asarray(arr, dtype=np.float32)[::s, ::s]
        if small.ndim == 3:
            small = small.mean(axis=-1)
        if not np.isfinite(small).all():
            self.nan = True
            small = np.nan_to_num(small)
        self.small.append(small)

    def stats(self) -> dict:
        cfg = self.cfg
        if not self.small or self.missing > self.total // 2:
            return {"alive": False, "reason": "no-output", "nan": self.nan,
                    "temporal_var": 0.0, "spatial_var": 0.0,
                    "frame_drop": self.missing}
        # Frames can vary in size if a node changes canvas — crop to smallest.
        h = min(f.shape[0] for f in self.small)
        w = min(f.shape[1] for f in self.small)
        stack = np.stack([f[:h, :w] for f in self.small])  # (T, h, w)

        mean_frame = stack.mean(axis=0)
        spatial_var = float(mean_frame.var())
        temporal_var = float(stack.var(axis=0).mean())

        # Mean consecutive-frame correlation — pure flicker decorrelates.
        corrs = []
        flat = stack.reshape(stack.shape[0], -1)
        for a, b in zip(flat[:-1], flat[1:]):
            sa, sb = a.std(), b.std()
            if sa < 1e-8 or sb < 1e-8:
                corrs.append(1.0)  # constant frames — "correlated", not flicker
            else:
                corrs.append(float(np.dot(a - a.mean(), b - b.mean())
                                   / (len(a) * sa * sb)))
        frame_corr = float(np.mean(corrs)) if corrs else 1.0

        reason = None
        if self.nan:
            reason = "nan"
        elif spatial_var < cfg.spatial_var_min:
            reason = "flat"
        elif temporal_var < cfg.temporal_var_min:
            reason = "static"
        elif temporal_var > cfg.flicker_var_min and frame_corr < cfg.flicker_corr_max:
            reason = "flicker"

        return {
            "alive": reason is None,
            "reason": reason,
            "nan": self.nan,
            "temporal_var": round(temporal_var, 6),
            "spatial_var": round(spatial_var, 6),
            "frame_corr": round(frame_corr, 4),
            "frame_drop": self.missing,
        }


def evaluate_frames(frames: list[np.ndarray | None],
                    cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Pure liveness classification of a frame stack (test hook)."""
    acc = LivenessAccumulator(cfg)
    for f in frames:
        acc.add(f)
    return acc.stats()


# ── Rendering ─────────────────────────────────────────────────────────


def render_genome(genome: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                  progress_cb: Callable[[str], None] | None = None) -> dict:
    """Render genome → output/sequences/shootout-<id>/output.mp4, fill
    genome['render'] + genome['liveness']. Never raises on node failures —
    executor errors become dead clips."""
    from image_pipeline.core.utils import set_canvas
    set_canvas(cfg.width, cfg.height)

    gid = genome["genome_id"]
    name = seq_name_for(gid)
    seq_dir = SEQUENCES_DIR / name
    work_dir = seq_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    graph = genome["graph"]
    # Copies: the executor and keyframe folding mutate node dicts.
    nodes = [dict(n, dirty=True) for n in graph.get("nodes", [])]
    edges = [dict(e) for e in graph.get("edges", [])]

    # Pin simulation length to the clip budget. The executor only overrides
    # n_frames when the param is present on the node; the sampler leaves it
    # off (frozen), so without this an Arch-A sim cooks its full default
    # (often 300+ frames) for a 96-frame clip.
    from image_pipeline.core.graph import get_all_node_defs
    defs = get_all_node_defs()
    for n in nodes:
        schema = (defs.get(n.get("method_id"), {}).get("params") or {})
        if "n_frames" in schema:
            n["params"] = {**(n.get("params") or {}), "n_frames": cfg.frames}
    seed = int(genome.get("seed", 42))

    executor = GraphExecutor(work_dir, fps=cfg.fps, in_memory=True,
                             audit_to_disk=False)
    acc = LivenessAccumulator(cfg)
    t0 = time.time()
    timed_out = False

    def _frame_gen():
        nonlocal timed_out
        for frame in range(cfg.frames):
            if time.time() - t0 > cfg.render_timeout_s:
                timed_out = True
                if progress_cb:
                    progress_cb(f"{gid}: timeout after {frame} frames")
                return
            arr = None
            try:
                flat, terminal_id, _errs = executor.execute(
                    nodes, edges, seed, frame=frame, frames=cfg.frames)
                render_id = next((n["id"] for n in nodes if n.get("render")), None)
                if render_id and render_id in flat:
                    terminal_id = render_id
                arr = (flat.get(terminal_id) or {}).get("image") if terminal_id else None
            except Exception as exc:  # cycle / unknown method — dead clip
                if progress_cb:
                    progress_cb(f"{gid}: frame {frame} error: {exc}")
            acc.add(arr)
            if arr is not None:
                yield np.asarray(arr, dtype=np.float32)
            if progress_cb and frame % 24 == 0:
                progress_cb(f"{gid}: frame {frame + 1}/{cfg.frames}")

    mp4_path = seq_dir / "output.mp4"
    out = None
    try:
        out = frames_to_mp4(_frame_gen(), mp4_path, fps=cfg.fps,
                            max_frames=cfg.frames)
    except Exception as exc:
        if progress_cb:
            progress_cb(f"{gid}: encode failed: {exc}")

    liveness = acc.stats()
    if timed_out:
        liveness = {**liveness, "alive": False, "reason": "timeout"}
    elif out is None and liveness.get("alive"):
        liveness = {**liveness, "alive": False, "reason": "encode-failed"}

    # The _work dir holds per-node transport PNGs from Arch-A sims — the mp4
    # is the durable artifact, so drop the scratch.
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        **genome,
        "render": {
            "seq_name": name,
            "mp4": f"/api/sequences/{name}/video.mp4",
            "frames": cfg.frames,
            "fps": cfg.fps,
            "w": cfg.width,
            "h": cfg.height,
            "wall_s": round(time.time() - t0, 1),
        },
        "liveness": liveness,
    }


def render_many(genomes: list[dict], cfg: ShootoutConfig = DEFAULT_CONFIG,
                progress_cb: Callable[[str], None] | None = None) -> list[dict]:
    """Render candidates concurrently (capped — plan §12 perf note).
    Returns genomes in input order with render/liveness filled."""
    results: dict[int, dict] = {}
    lock = threading.Lock()

    def _safe_progress(msg: str) -> None:
        if progress_cb:
            with lock:
                progress_cb(msg)

    with ThreadPoolExecutor(max_workers=cfg.render_concurrency) as ex:
        futs = {ex.submit(render_genome, g, cfg, _safe_progress): i
                for i, g in enumerate(genomes)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                _safe_progress(f"{genomes[i]['genome_id']}: render crashed: {exc}")
                results[i] = {**genomes[i],
                              "render": None,
                              "liveness": {"alive": False, "reason": "crash",
                                           "error": str(exc)}}
    return [results[i] for i in range(len(genomes))]
