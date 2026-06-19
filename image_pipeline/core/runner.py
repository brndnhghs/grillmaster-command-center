"""
Parallel/sequential runner with progress tracking, timing, and caching.
"""
from __future__ import annotations
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import cache
from .registry import MethodMeta, timed_run

# Serialises the seed-then-use sequence so concurrent threads cannot race on
# the global random/numpy RNG state that each method resets via seed_all().
_rng_lock = threading.Lock()


def run_sequential(
    metas: list[MethodMeta],
    out_dir: Path,
    seed: int,
    force: bool = False,
    params: dict | None = None,
    progress_cb: Callable | None = None,
) -> list[tuple[MethodMeta, float, bool]]:
    """Run methods one at a time. Returns (meta, elapsed_seconds, from_cache)."""
    results: list[tuple[MethodMeta, float, bool]] = []
    total = len(metas)

    for idx, meta in enumerate(metas):
        label = f"[{idx+1}/{total}] {meta.id} {meta.name}"

        # Check cache
        cached = cache.exists(meta.id, seed, out_dir, params) if not force else None
        if cached:
            if progress_cb:
                progress_cb(label, meta, 0, True)
            results.append((meta, 0.0, True))
            continue

        if progress_cb:
            progress_cb(label, meta, 0, False)

        start = time.time()
        try:
            meta.fn(out_dir, seed, params=params)
        except TypeError:
            meta.fn(out_dir, seed)
        elapsed = time.time() - start

        # Store in cache
        out_path = out_dir / meta.filename()
        if out_path.exists():
            cache.store(meta.id, seed, out_path, params)

        if progress_cb:
            progress_cb(label, meta, elapsed, False)
        results.append((meta, elapsed, False))

    return results


def run_parallel(
    metas: list[MethodMeta],
    out_dir: Path,
    seed: int,
    max_workers: int = 4,
    force: bool = False,
    params: dict | None = None,
    progress_cb: Callable | None = None,
) -> list[tuple[MethodMeta, float, bool]]:
    """Run methods in parallel using a thread pool."""
    results: list[tuple[MethodMeta, float, bool] | None] = [None] * len(metas)
    total = len(metas)
    completed = 0

    def run_one(meta: MethodMeta, idx: int):
        nonlocal completed
        label = f"[{idx+1}/{total}] {meta.id} {meta.name}"

        cached = cache.exists(meta.id, seed, out_dir, params) if not force else None
        if cached:
            if progress_cb:
                progress_cb(label, meta, 0, True)
            return (meta, 0.0, True)

        if progress_cb:
            progress_cb(label, meta, 0, False)

        with _rng_lock:
            start = time.time()
            try:
                meta.fn(out_dir, seed, params=params)
            except TypeError:
                meta.fn(out_dir, seed)
            elapsed = time.time() - start

        out_path = out_dir / meta.filename()
        if out_path.exists():
            cache.store(meta.id, seed, out_path, params)

        if progress_cb:
            progress_cb(label, meta, elapsed, False)
        return (meta, elapsed, False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, meta, idx): idx for idx, meta in enumerate(metas)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                meta = metas[idx]
                print(f"\n  ✗ {meta.id} {meta.name}: {e}")
                results[idx] = (meta, 0.0, False)

    return [r for r in results if r is not None]


def default_progress(label: str, meta: MethodMeta, elapsed: float, from_cache: bool):
    """Default progress callback used by the CLI."""
    if from_cache:
        print(f"  ⊛ {label}  (cached)")
    elif elapsed > 0:
        print(f"  ✓ {label}  ({elapsed:.1f}s)")
    else:
        print(f"  → {label}")