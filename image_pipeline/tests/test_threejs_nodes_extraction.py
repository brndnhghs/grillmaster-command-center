"""3D-node-defs extraction regression test (ROADMAP R8 / TD-07).

Pins the extraction of the three.js 3D node definitions out of
``core/graph.py`` into ``core/threejs_nodes.py`` (TD-07). The defs are pure
serialisable metadata consumed by ``graph.get_all_node_defs()``; they carry no
execution logic, so the one-way dependency (graph imports threejs_nodes) must
stay a pure re-export with identical content.

This test guards against:
  1. The 3D defs silently vanishing from ``get_all_node_defs()``.
  2. Structural drift between ``threejs_nodes.THREEJS_3D_NODE_DEFS`` and what
     ``get_all_node_defs()`` actually returns (a re-inline or partial edit).
  3. The backward-compat alias ``_THREEJS_3D_NODE_DEFS`` (still imported by
     server.py and test_3d_sidecar_render.py) disappearing from graph.py.
"""

from __future__ import annotations

import image_pipeline.core.graph as graph
import image_pipeline.core.threejs_nodes as threejs
from image_pipeline.core.graph import get_all_node_defs


EXPECTED_3D_IDS = {
    "__geometry__", "__material__", "__mesh3d__", "__group3d__",
    "__light3d__", "__camera3d__", "__scene_render__", "__scene3d__",
    "__gltf__", "__usd__",
}


def test_threejs_defs_module_exists_and_exports():
    assert hasattr(threejs, "THREEJS_3D_NODE_DEFS")
    assert hasattr(threejs, "THREEJS_POSTFX_PARAMS")
    assert hasattr(threejs, "MODEL_PLACEMENT_PARAMS")
    assert set(threejs.THREEJS_3D_NODE_DEFS) == EXPECTED_3D_IDS


def test_graph_still_exposes_3d_defs_via_get_all_node_defs():
    defs = get_all_node_defs()
    present = {k for k in defs if k in EXPECTED_3D_IDS}
    assert present == EXPECTED_3D_IDS, f"missing 3D node defs: {EXPECTED_3D_IDS - present}"


def test_3d_defs_match_source_module():
    """get_all_node_defs() must return byte-equivalent 3D defs to the module."""
    defs = get_all_node_defs()
    for mid, src in threejs.THREEJS_3D_NODE_DEFS.items():
        got = defs[mid]
        for fld in ("method_id", "name", "category", "inputs", "outputs",
                    "params", "deprecated", "description", "is_time_varying"):
            assert src.get(fld) == got.get(fld), (
                f"{mid}.{fld} drifted after extraction from core/threejs_nodes.py"
            )


def test_backward_compat_alias_present_in_graph():
    # server.py and test_3d_sidecar_render.py import _THREEJS_3D_NODE_DEFS
    # from graph; the extraction must preserve that alias.
    assert hasattr(graph, "_THREEJS_3D_NODE_DEFS")
    assert graph._THREEJS_3D_NODE_DEFS is threejs.THREEJS_3D_NODE_DEFS


def test_graph_no_longer_defines_helper_inline():
    # The helper + inline dicts were moved out; graph.py should only import them.
    assert not hasattr(graph, "_threejs_node_def"), (
        "extraction incomplete — _threejs_node_def still defined in graph.py"
    )
