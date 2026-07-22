"""Regression: node 437 (Marbling) emitted a degenerate uint8 image.

Promoted from captured node issue ni-0001 (2026-07-22T07:49:57+00:00).
The pipeline is deterministic, so this is the exact input that failed.
"""
import tempfile
from pathlib import Path

import numpy as np

import image_pipeline.methods  # noqa: F401
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas

NODES = [{'id': 'n',
  'method_id': '437',
  'params': {'anim_speed': 1.0,
             'bg_mode': 'gradient',
             'ink_palette': 'jewel',
             'n_drops': 2,
             'n_tines': 18,
             'ragged': 0.5,
             'ring_gap': 16,
             'rings': 12,
             'tine_spacing': 28,
             'tine_strength': 2.0},
  'render': True}]
EDGES = []


def test_ni_0001_node_437_is_healthy():
    set_canvas(256, 192)
    ex = GraphExecutor(Path(tempfile.mkdtemp()), in_memory=True, audit_to_disk=False)
    flat, _t, errs = ex.execute(nodes=NODES, edges=EDGES, seed=7,
                                frame=1, frames=8)
    assert not errs, f"node raised: {errs}"
    img = flat.get('n', {}).get("image")
    assert img is not None, "node produced no image"
    arr = np.asarray(img, dtype=np.float32)
    assert np.isfinite(arr).all(), "output contains NaN/Inf"
    assert arr.min() != arr.max(), "output is a constant image"

