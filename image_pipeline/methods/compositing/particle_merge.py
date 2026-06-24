"""Particle Merge — combines two PARTICLES wires into a single stream."""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ...core.registry import method
from ...core.utils import save, mn, write_particles, W, H


@method(
    id="140",
    name="Particle Merge",
    category="compositing",
    tags=["particles", "merge", "combine"],
    inputs={"particles_a": "PARTICLES", "particles_b": "PARTICLES"},
    outputs={"particles": "PARTICLES"},
    params={
        "mode": {
            "description": "merge mode",
            "default": "concat",
            "choices": ["concat", "interleave", "a_only", "b_only"],
        },
    }
)
def method_particle_merge(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}
    particles_a_path = params.get("particles_a_path", "")
    particles_b_path = params.get("particles_b_path", "")
    mode = params.get("mode", "concat")

    parts_a = (
        np.load(particles_a_path).astype(np.float32)
        if particles_a_path and os.path.exists(particles_a_path)
        else None
    )
    parts_b = (
        np.load(particles_b_path).astype(np.float32)
        if particles_b_path and os.path.exists(particles_b_path)
        else None
    )

    if mode == "a_only":
        merged = parts_a
    elif mode == "b_only":
        merged = parts_b
    elif mode == "interleave" and parts_a is not None and parts_b is not None:
        cols = max(parts_a.shape[1], parts_b.shape[1])
        n = min(len(parts_a), len(parts_b))
        inter = np.zeros((2 * n, cols), dtype=np.float32)
        inter[0::2, : parts_a.shape[1]] = parts_a[:n]
        inter[1::2, : parts_b.shape[1]] = parts_b[:n]
        tail = [inter]
        if len(parts_a) > n:
            p = np.zeros((len(parts_a) - n, cols), dtype=np.float32)
            p[:, : parts_a.shape[1]] = parts_a[n:]
            tail.append(p)
        if len(parts_b) > n:
            p = np.zeros((len(parts_b) - n, cols), dtype=np.float32)
            p[:, : parts_b.shape[1]] = parts_b[n:]
            tail.append(p)
        merged = np.concatenate(tail, axis=0)
    else:
        candidates = [p for p in [parts_a, parts_b] if p is not None]
        merged = np.concatenate(candidates, axis=0) if candidates else None

    if merged is None or len(merged) == 0:
        save(np.zeros((H, W, 3), dtype=np.float32), mn(140, "Particle Merge"), out_dir)
        return

    write_particles(out_dir, merged)

    n_a = len(parts_a) if parts_a is not None else 0
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    xs = np.clip(merged[:, 0], 0, W - 1).astype(int)
    ys = np.clip(merged[:, 1], 0, H - 1).astype(int)

    if n_a > 0:
        a_mask = np.arange(len(merged)) < n_a
        np.add.at(canvas[:, :, 0], (ys[a_mask], xs[a_mask]), 0.2)
        np.add.at(canvas[:, :, 1], (ys[a_mask], xs[a_mask]), 0.4)
        np.add.at(canvas[:, :, 2], (ys[a_mask], xs[a_mask]), 1.0)
        b_mask = ~a_mask
        np.add.at(canvas[:, :, 0], (ys[b_mask], xs[b_mask]), 1.0)
        np.add.at(canvas[:, :, 1], (ys[b_mask], xs[b_mask]), 0.5)
        np.add.at(canvas[:, :, 2], (ys[b_mask], xs[b_mask]), 0.1)
    else:
        np.add.at(canvas[:, :, 0], (ys, xs), 0.3)
        np.add.at(canvas[:, :, 1], (ys, xs), 0.6)
        np.add.at(canvas[:, :, 2], (ys, xs), 1.0)

    canvas = np.clip(canvas, 0.0, 1.0)
    vis = Image.fromarray((canvas * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1))
    save(np.array(vis, dtype=np.float32) / 255.0, mn(140, "Particle Merge"), out_dir)
