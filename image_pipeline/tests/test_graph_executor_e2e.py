"""End-to-end tests for GraphExecutor — validates the full execution pipeline."""
from __future__ import annotations
import numpy as np
import pytest
from image_pipeline.core.graph import GraphExecutor, GraphError
import image_pipeline.methods  # noqa: F401

def _make_node(node_id, method_id, render=False, params=None, dirty=True):
    return {'id': node_id, 'method_id': method_id, 'params': params or {},
            'render': render, 'dirty': dirty, 'x': 0, 'y': 0,
            'start_frame': 0, 'end_frame': 0, 'keyframes': [],
            'paramKeyframes': {}, 'prebake': 0}

def _make_edge(src, dst, src_port='image', dst_port='image_in', feedback=False):
    return {'src_node': src, 'src_port': src_port, 'dst_node': dst,
            'dst_port': dst_port, 'feedback': feedback}

def test_single_generator(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '05', render=True)]
    outputs, terminal_id, errors = executor.execute(nodes, [], seed=42)
    assert terminal_id == 'n1' and 'n1' in outputs
    img = outputs['n1']['image']
    assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert 'luminance' in outputs['n1']

def test_two_node_graph(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '05'), _make_node('n2', '13', render=True)]
    edges = [_make_edge('n1', 'n2')]
    outputs, tid, err = executor.execute(nodes, edges, seed=42)
    assert tid == 'n2' and 'n1' in outputs and 'n2' in outputs
    assert outputs['n2']['image'].shape == outputs['n1']['image'].shape

def test_dirty_flag_skip(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '05', render=True, dirty=True)]
    o1, _, _ = executor.execute(nodes, [], seed=42)
    nodes[0]['dirty'] = False
    o2, _, _ = executor.execute(nodes, [], seed=42)
    # Tolerate PNG round-trip quantization (uint8 → float32 loses ~1/255)
    np.testing.assert_allclose(o1['n1']['image'], o2['n1']['image'], atol=0.005)

def test_dirty_flag_recook(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '05', render=True, dirty=True)]
    o1, _, _ = executor.execute(nodes, [], seed=42)
    o2, _, _ = executor.execute(nodes, [], seed=42)
    np.testing.assert_array_equal(o1['n1']['image'], o2['n1']['image'])

def test_unknown_method_raises(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '99999', render=True)]
    with pytest.raises(GraphError, match='Unknown method'):
        executor.execute(nodes, [], seed=42)

def test_topological_sort(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '05'), _make_node('n2', '13'), _make_node('n3', '77', render=True)]
    edges = [_make_edge('n1','n2'), _make_edge('n2','n3')]
    outputs, tid, _ = executor.execute(nodes, edges, seed=42)
    assert all(n in outputs for n in ['n1','n2','n3'])

def test_branching_graph(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1','05'), _make_node('n2','77'), _make_node('n3','138', render=True)]
    edges = [_make_edge('n1','n3'), _make_edge('n2','n3')]
    outputs, tid, _ = executor.execute(nodes, edges, seed=42)
    assert all(n in outputs for n in ['n1','n2','n3'])

def test_cycle_detection(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1','05'), _make_node('n2','13', render=True)]
    edges = [_make_edge('n1','n2'), _make_edge('n2','n1')]
    with pytest.raises(GraphError, match='cycle'):
        executor.execute(nodes, edges, seed=42)

def test_architecture_a_simulation_cache(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1','32', render=True, params={'n_frames': 4})]
    o1, _, _ = executor.execute(nodes, [], seed=42, frame=0, frames=4)
    o2, _, _ = executor.execute(nodes, [], seed=42, frame=0, frames=4)
    np.testing.assert_array_equal(o1['n1']['image'], o2['n1']['image'])

def test_feedback_edge(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1','05', render=True)]
    edges = [_make_edge('n1','n1','luminance','scale',feedback=True)]
    outputs, tid, _ = executor.execute(nodes, edges, seed=42, frame=0)
    assert tid == 'n1' and outputs['n1']['image'] is not None

def test_multi_frame_sequence(tmp_path):
    executor = GraphExecutor(tmp_path)
    nodes = [_make_node('n1', '05', render=True)]
    frames = [executor.execute(nodes, [], seed=42, frame=f, frames=5)[0]['n1']['image']
              for f in range(5)]
    assert len(frames) == 5
    # All frames should have valid pixel range
    for img in frames:
        assert img.min() >= 0.0 and img.max() <= 1.0
