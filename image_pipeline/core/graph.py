"""Node graph system — wires registered methods into a DAG and executes it."""
from __future__ import annotations

import logging
import math
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from . import registry
from .expr import eval_param
from .port_types import all_port_types  # noqa: F401 — ensures registry loads
from .timeline import Timeline, make_timeline, KeyframeTrack
from .utils import W, H


# ── Node / edge schema ────────────────────────────────────────────────

@dataclass
class NodeDef:
    method_id:   str
    inputs:      dict[str, str]   # port_name -> port type string
    outputs:     dict[str, str]
    param_ports: set[str] = field(default_factory=set)  # input ports that map to params


@dataclass
class GraphEdge:
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str
    feedback: bool = False


@dataclass
class GraphNode:
    id:          str
    method_id:   str = ""  # empty for group nodes
    params:      dict[str, Any] = field(default_factory=dict)
    x:           float = 0.0
    y:           float = 0.0
    render:      bool  = False
    dirty:       bool  = True
    start_frame: int   = 0
    end_frame:   int   = 0
    keyframes:   list[dict] = field(default_factory=list)
    paramKeyframes: dict[str, list[dict]] = field(default_factory=dict)
    prebake:     int   = 0


# ── Auto-generate NodeDefs from registry ─────────────────────────────

def _make_node_def(meta: registry.MethodMeta) -> NodeDef:
    _PORT_TYPE_MAP = {
        "IMAGE": "image", "SCALAR": "scalar",
        "FIELD": "field", "PARTICLES": "particles",
        "MASK": "mask", "ANY": "any",
    }
    outputs: dict[str, str] = {
        k: _PORT_TYPE_MAP.get(v.upper(), "any")
        for k, v in meta.outputs.items()
    }

    # Inputs: image_in (feeds input_image) + one SCALAR/FIELD port per wireable param.
    # SCALAR ports: int or float defaults only (bool excluded — subclass of int but not wireable).
    # FIELD  ports: list or tuple defaults only.
    # str defaults (categorical choices) and bool defaults are intentionally excluded.
    # Nodes with explicit inputs=None skip image_in — they own their own port declarations.
    # Nodes with inputs={} also skip image_in (no wireable inputs at all).
    # Nodes with inputs=None (default) get the auto-generated image_in port.
    inputs: dict[str, str] = {}
    if meta.inputs is None:
        # Default: auto-generate image_in
        inputs["image_in"] = "image"
    elif meta.inputs:
        # Explicitly declared inputs — use them directly
        for port_name, type_str in meta.inputs.items():
            inputs[port_name] = _PORT_TYPE_MAP.get(type_str.upper(), "any")
    # else: inputs={} — no inputs at all (pure data source like Timeline)

    # ── Auto-detect wireable param ports ──────────────────────────────
    # Only add params that don't already have explicit input declarations
    # and don't have min/max constraints (internal sliders, not wireable)
    param_ports: set[str] = set()
    declared_inputs = set(meta.inputs or {})
    for pname, spec in (meta.params or {}).items():
        if pname in declared_inputs:
            continue  # already explicitly declared
        if not isinstance(spec, dict):
            continue
        if 'min' in spec or 'max' in spec:
            continue  # has slider constraints — internal control, not wireable
        default = spec.get("default")
        if default is None:
            continue
        if isinstance(default, bool):
            continue  # bool is a subclass of int — exclude before the int/float check
        if isinstance(default, str):
            continue  # categorical choices aren't wireable
        if isinstance(default, (int, float)):
            inputs[pname] = "scalar"   # SCALAR: receives a numeric scalar output (NOT luminance — that's a (H,W) FIELD)
            param_ports.add(pname)
        elif isinstance(default, (list, tuple)):
            inputs[pname] = "field"    # FIELD: receives image ndarray
            param_ports.add(pname)

    return NodeDef(method_id=meta.id, inputs=inputs, outputs=outputs, param_ports=param_ports)


def _threejs_node_def(method_id: str, name: str, *,
                      category: str = "client_3d",
                      tags: list[str] | None = None,
                      inputs: dict[str, str] | None = None,
                      outputs: dict[str, str] | None = None,
                      params: dict[str, dict] | None = None,
                      description: str = "",
                      deprecated: bool = False) -> dict:
    """Build a serialisable NodeDef dict for a three.js 3D node."""
    return {
        "method_id": method_id, "name": name,
        "category": category, "tags": tags or [],
        "inputs": inputs or {}, "outputs": outputs or {},
        "param_ports": [], "description": description,
        "version": 1, "deprecated": deprecated,
        "start_frame": 0, "end_frame": 0, "prebake": 0,
        "params": params or {},
    }


_THREEJS_3D_NODE_DEFS: dict[str, dict] = {
    '__geometry__': _threejs_node_def('__geometry__', '3D Geometry', tags=['3d', 'client'],
        outputs={"geometry": "geometry"},
        description='Emits a geometry for a Mesh node.',
        params={
            "shape":  {"description": "geometry shape",
                "choices": ["box","sphere","torus","torusknot","cone","cylinder",
                            "icosahedron","dodecahedron","plane"],
                "default": "torusknot"},
            "size":   {"description": "overall size", "min": 0.1, "max": 3, "default": 1},
            "detail": {"description": "tessellation detail", "min": 0, "max": 1, "default": 0.5},
        }),
    '__material__': _threejs_node_def('__material__', '3D Material', tags=['3d', 'client', 'pbr'],
        outputs={"material": "material"},
        description='PBR material for a Mesh node.',
        params={
            "color":              {"description": "base color", "default": "#4a9eff"},
            "metalness":          {"description": "metalness", "min": 0, "max": 1, "default": 0.4},
            "roughness":          {"description": "roughness", "min": 0, "max": 1, "default": 0.35},
            "emissive":           {"description": "emissive color", "default": "#000000"},
            "emissive_intensity": {"description": "emissive intensity", "min": 0, "max": 4, "default": 1},
            "flat_shading":       {"description": "flat shading (0/1)", "min": 0, "max": 1, "default": 0},
        }),
    '__mesh3d__': _threejs_node_def('__mesh3d__', '3D Mesh', tags=['3d', 'client'],
        inputs={"geometry": "geometry", "material": "material"},
        outputs={"object": "object3d"},
        description='Geometry + Material → transformable object (keyframeable).',
        params={
            "pos_x": {"description": "position X", "min": -5, "max": 5, "default": 0},
            "pos_y": {"description": "position Y", "min": -5, "max": 5, "default": 0},
            "pos_z": {"description": "position Z", "min": -5, "max": 5, "default": 0},
            "rot_x": {"description": "rotation X (deg)", "min": -180, "max": 180, "default": 0},
            "rot_y": {"description": "rotation Y (deg)", "min": -180, "max": 180, "default": 0},
            "rot_z": {"description": "rotation Z (deg)", "min": -180, "max": 180, "default": 0},
            "spin_speed": {"description": "auto Y-spin (rad/s)", "min": 0, "max": 4, "default": 0.6},
            "scale": {"description": "uniform scale", "min": 0.1, "max": 3, "default": 1},
        }),
    '__group3d__': _threejs_node_def('__group3d__', '3D Group', tags=['3d', 'client'],
        inputs={"object_a": "object3d", "object_b": "object3d"},
        outputs={"object": "object3d"},
        description='Combine two objects into one.'),
    '__light3d__': _threejs_node_def('__light3d__', '3D Light', tags=['3d', 'client'],
        outputs={"light": "light"},
        description='Light for a Scene Render node.',
        params={
            "type":      {"description": "light type",
                "choices": ["point","directional","spot"], "default": "point"},
            "pos_x":     {"description": "position X", "min": -10, "max": 10, "default": 3},
            "pos_y":     {"description": "position Y", "min": -10, "max": 10, "default": 4},
            "pos_z":     {"description": "position Z", "min": -10, "max": 10, "default": 5},
            "color":     {"description": "light color", "default": "#ffffff"},
            "intensity": {"description": "intensity", "min": 0, "max": 500, "default": 60},
        }),
    '__camera3d__': _threejs_node_def('__camera3d__', '3D Camera', tags=['3d', 'client'],
        outputs={"camera": "camera"},
        description='Camera for a Scene Render node.',
        params={
            "pos_x": {"description": "camera X", "min": -12, "max": 12, "default": 0},
            "pos_y": {"description": "camera Y", "min": -12, "max": 12, "default": 0},
            "pos_z": {"description": "camera Z (dolly)", "min": 0.5, "max": 16, "default": 4},
            "look_x": {"description": "look-at X", "min": -5, "max": 5, "default": 0},
            "look_y": {"description": "look-at Y", "min": -5, "max": 5, "default": 0},
            "look_z": {"description": "look-at Z", "min": -5, "max": 5, "default": 0},
            "fov":    {"description": "field of view", "min": 15, "max": 110, "default": 50},
        }),
    '__scene_render__': _threejs_node_def('__scene_render__', '3D Scene Render',
        tags=['3d', 'client', 'render'],
        inputs={"object": "object3d", "light": "light", "camera": "camera"},
        outputs={"image": "image", "luminance": "field"},
        description='Assemble object(s) + light + camera → rendered image.',
        params={
            "bg_color": {"description": "background color", "default": "#0a0e18"},
            "ambient":  {"description": "ambient light", "min": 0, "max": 1, "default": 0.35},
        }),
    '__scene3d__': _threejs_node_def('__scene3d__', '3D Scene (legacy)',
        tags=['3d', 'client', 'deprecated'],
        inputs={"object": "object3d", "light": "light", "camera": "camera"},
        outputs={"image": "image", "luminance": "field"},
        description='Legacy 3D scene node (deprecated, use Scene Render).',
        deprecated=True,
        params={
            "bg_color": {"description": "background color", "default": "#0a0e18"},
            "ambient":  {"description": "ambient light", "min": 0, "max": 1, "default": 0.35},
        }),
    '__gltf__': _threejs_node_def('__gltf__', '3D Model (GLTF)',
        tags=['3d', 'client', 'gltf'],
        outputs={"object": "object3d"},
        description='Load a .gltf/.glb model as an object.',
        params={
            "url":        {"description": "model URL (.gltf/.glb)",
                "default": "https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf"},
            "scale":      {"description": "uniform scale", "min": 0.05, "max": 5, "default": 1},
            "spin_speed": {"description": "auto Y-spin (rad/s)", "min": 0, "max": 4, "default": 0.6},
        }),
}

@cache
def get_all_node_defs() -> dict[str, dict]:
    """Return serialisable NodeDef dict keyed by method_id.
    Merges Python-registered methods with three.js 3D node definitions.
    Cached for performance — call clear_node_defs_cache() after hot-reload."""
    result = {**_THREEJS_3D_NODE_DEFS}
    for mid, meta in registry.get_all().items():
        nd = _make_node_def(meta)
        result[mid] = {
            "method_id":   nd.method_id,
            "name":        meta.name,
            "category":    meta.category,
            "tags":        meta.tags,
            "params":      meta.params,
            "inputs":      dict(nd.inputs),
            "outputs":     dict(nd.outputs),
            "param_ports": list(nd.param_ports),
            "description": meta.description,
            "version":     meta.version,
            "deprecated":  meta.deprecated,
            "start_frame": 0,
            "end_frame":   0,
            "prebake":     0,
        }
    return result


def clear_node_defs_cache() -> None:
    """Invalidate the cached node defs after hot-reload so the next request
    rebuilds from the updated registry."""
    get_all_node_defs.cache_clear()


# ── Var-injection helpers ─────────────────────────────────────────────

# Synonym table for name-based param scoring (src_port → set of matching param name fragments)
_SYNONYMS: dict[str, set[str]] = {
    "luminance": {"brightness", "value", "intensity", "luma", "light"},
    "field":     {"grid", "map", "array", "data"},
    "speed":     {"velocity", "rate"},
    "frequency": {"freq", "rate", "hz"},
}


def _score_param(src_port: str, param_names: list[str]) -> str | None:
    """Return the highest-scoring param name for src_port by name similarity.

    Scoring: exact=10, synonym=5, substring=2, no match=0.
    Returns None if no param scores above zero.
    """
    if not param_names:
        return None
    src_lower = src_port.lower()
    synonyms_for_src = _SYNONYMS.get(src_lower, set()) | {src_lower}
    best: str | None = None
    best_score = 0
    for pname in param_names:
        pname_lower = pname.lower()
        if pname_lower == src_lower:
            score = 10
        elif pname_lower in synonyms_for_src or src_lower in _SYNONYMS.get(pname_lower, set()):
            score = 5
        elif src_lower in pname_lower or pname_lower in src_lower:
            score = 2
        else:
            score = 0
        if score > best_score:
            best, best_score = pname, score
    return best


def _eligible_params(params: dict, src_type: str) -> list[tuple[str, dict]]:
    """Return params eligible for injection given the source wire type.

    SCALAR (numeric output) → params with int or float defaults (not bool, not str).
    FIELD  (image ndarray)   → params with list or tuple defaults.
    NOTE: ``luminance`` is a per-pixel FIELD (H,W), never injected as a scalar param.
    """
    out: list[tuple[str, dict]] = []
    for k, spec in params.items():
        if not isinstance(spec, dict):
            continue
        default = spec.get("default")
        if src_type == "scalar":
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                out.append((k, spec))
        elif src_type == "field":
            if isinstance(default, (list, tuple)):
                out.append((k, spec))
    return out


def _inject_typed(
    run_params: dict, param: str, value: Any, src_type: str, node_params: dict
) -> None:
    """Write value into run_params[param] with type-safe coercion.

    SCALAR → int param: round(); SCALAR → float param: pass as-is.
    FIELD  → list/tuple param: pass raw ndarray.
    Logs a warning and skips on type mismatch.
    """
    orig = node_params.get(param)
    if src_type == "scalar":
        if isinstance(orig, bool):
            logging.warning("graph: SCALAR wire to bool param %r skipped", param)
            return
        if orig is None or isinstance(orig, (int, float)):
            val = round(float(value)) if isinstance(orig, int) else float(value)
            run_params[param] = val
            # Also inject as uniform field for FIELD-input methods.
            # broadcast_to is a zero-copy read-only view — a np.full here
            # allocated ~1.5 MB per scalar per node per frame whether or not
            # the method ever read _field_<param>. Consumers only use these
            # in arithmetic; anything needing a writable buffer must copy.
            run_params[f"_field_{param}"] = np.broadcast_to(
                np.float32(val), (int(H), int(W))
            )
        else:
            logging.warning(
                "graph: SCALAR→%s type mismatch for param %r, skipping",
                type(orig).__name__, param,
            )
    elif src_type == "field":
        # FIELD can wire to list/tuple params (old convention) OR
        # to int/float params (new convention — method reads _field_<name>)
        if isinstance(orig, (list, tuple)):
            run_params[param] = value
        elif orig is None or isinstance(orig, (int, float)):
            # Inject as _field_<param> so method can read it
            run_params[f"_field_{param}"] = value
        else:
            logging.warning(
                "graph: FIELD→%s type mismatch for param %r, skipping",
                type(orig).__name__, param,
            )


# ── Per-param keyframe evaluation ─────────────────────────────────────


def _evaluate_param_track(keyframes: list[dict], frame: int) -> Any | None:
    """Evaluate a single param's keyframe track at a given frame.

    keyframes is a sorted list of {frame, value, easing?, handle_in?, handle_out?}.
    Returns the interpolated value, or None if no keyframes exist.
    """
    if not keyframes:
        return None

    # Before first keyframe — hold
    if frame <= keyframes[0]["frame"]:
        return keyframes[0]["value"]

    # After last keyframe — hold
    if frame >= keyframes[-1]["frame"]:
        return keyframes[-1]["value"]

    # Find the segment containing this frame
    for i in range(len(keyframes) - 1):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]
        if kf_a["frame"] <= frame < kf_b["frame"]:
            window = kf_b["frame"] - kf_a["frame"]
            if window <= 0:
                return kf_b["value"]
            t = (frame - kf_a["frame"]) / window
            easing = kf_b.get("easing", "linear")
            from .easing import apply_easing
            t_eased = apply_easing(t, easing,
                                   handle_in=kf_b.get("handle_in"),
                                   handle_out=kf_b.get("handle_out"))
            a_val = kf_a["value"]
            b_val = kf_b["value"]
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                return a_val + (b_val - a_val) * t_eased
            # Non-numeric: snap at midpoint
            return a_val if t_eased < 0.5 else b_val

    return None


# ── Incremental-recook helpers ────────────────────────────────────────


def _compute_live_dirty(
    nodes: list[dict],
    edges: list[dict],
    initially_dirty: set[str],
) -> set[str]:
    """Propagate an initial dirty set forward through the topology.

    Returns the full set of node IDs that must re-cook this frame:
    the initial set PLUS every node that is transitively downstream of
    any initially-dirty node.  Feedback edges are excluded from cascading
    (they carry the *previous* frame's output and break the DAG cycle).
    """
    dirty = set(initially_dirty)
    # Build adjacency (non-feedback only)
    downstream: dict[str, set[str]] = {n["id"]: set() for n in nodes}
    for e in edges:
        if not e.get("feedback", False):
            downstream.setdefault(e["src_node"], set()).add(e["dst_node"])

    # BFS forward
    queue = list(dirty)
    while queue:
        nid = queue.pop()
        for child in downstream.get(nid, ()):
            if child not in dirty:
                dirty.add(child)
                queue.append(child)
    return dirty


# ── Graph executor ────────────────────────────────────────────────────

class GraphError(Exception):
    pass


def _stable_node_offset(node_id: str) -> int:
    """Deterministic per-node seed offset.

    Built-in hash() is randomized per process (PYTHONHASHSEED), which made
    node seeds — and therefore output — change across server restarts.
    """
    import hashlib
    return int.from_bytes(hashlib.sha1(node_id.encode()).digest()[:2], "big")


# Per-frame clock/context keys injected by the executor and the live loop.
# They are NOT part of a simulation's identity, so excluding them keeps the
# Architecture-A sim cache stable across frames — otherwise the live loop's
# per-frame `time = float(frame)` changes the key every frame and the cook
# is repeated on every frame instead of served from cache.
_VOLATILE_PARAM_KEYS = frozenset({
    "time", "frame", "frame_seed", "_timeline", "_input_image", "input_image",
})


def _node_params_hash(method_id: str, params: dict) -> str:
    """Stable digest of a node's identity + defining params for the sim cache.

    `method_id` MUST be part of the key: swapping one sim node for another on
    the same node_id (same id, often identical default params like
    {"n_frames": 8}) must never serve the previously cooked node's frames.
    """
    import json as _json
    return _json.dumps(
        {"__method_id__": method_id,
         **{k: str(v) for k, v in sorted(params.items())
            if k not in _VOLATILE_PARAM_KEYS}},
        sort_keys=True,
    )


def _write_error_placeholder(node_dir: Path, write: bool = True) -> np.ndarray:
    """Dark-red W×H placeholder as float32 ndarray [0,1]; PNG written only when
    `write` is set (render/audit mode) — live mode keeps the hot path disk-free."""
    arr_u8 = np.full((H, W, 3), [58, 0, 0], dtype=np.uint8)
    if write:
        from PIL import Image as _PILe
        _PILe.fromarray(arr_u8).save(str(node_dir / "0000_error.png"))
    return arr_u8.astype(np.float32) / 255.0


class GraphExecutor:
    """Execute a node graph for one or more frames."""

    # Byte budget for the Arch-A simulation cache. A 300-frame sim at 768×512
    # float32 RGB is ~1.4 GB per node; without a cap the process grows without
    # bound (BUG-8a). Oldest entries are evicted first.
    SIM_CACHE_MAX_BYTES = 1_500_000_000

    def __init__(self, out_dir: Path, fps: int = 24, in_memory: bool = False,
                 audit_to_disk: bool = True):
        self.out_dir = out_dir
        self._fps = fps
        self._in_memory = in_memory
        # audit_to_disk=False (live mode) skips every per-node PNG/sidecar
        # write — the in-memory payload bus is the only transport. Render
        # jobs keep the default True so the on-disk audit trail survives.
        self.audit_to_disk = audit_to_disk
        # Keyed by node_id → {"image": ndarray | None, "luminance": float}
        self._prev_outputs: dict[str, dict[str, Any]] = {}
        # Simulation state cache: keyed by (node_id, seed) → list of ndarray frames
        self._sim_cache: dict[tuple[str, int], list] = {}
        self._sim_params_hash: dict[str, int] = {}
        # Cached sub-executors for group nodes, keyed by group node id.
        # A fresh executor per frame would lose feedback state and re-run
        # Arch-A sims inside groups from scratch every frame (BUG-6).
        self._group_executors: dict[str, "GraphExecutor"] = {}
        # Diagnostics for the last completed frame — written by execute(), read by server
        self.last_frame_stats: dict = {}

    def _evict_sim_cache(self) -> None:
        """Drop oldest sim-cache entries until under the byte budget."""
        def _entry_bytes(frames: list) -> int:
            total = 0
            for f in frames:
                a = f.get("image") if isinstance(f, dict) else f
                if isinstance(a, np.ndarray):
                    total += a.nbytes
            return total

        total = sum(_entry_bytes(v) for v in self._sim_cache.values())
        while total > self.SIM_CACHE_MAX_BYTES and len(self._sim_cache) > 1:
            oldest_key = next(iter(self._sim_cache))
            total -= _entry_bytes(self._sim_cache.pop(oldest_key))
            self._sim_params_hash.pop(oldest_key[0], None)

    def execute(
        self,
        nodes: list[dict],
        edges: list[dict],
        seed: int,
        frame: int = 0,
        frames: int = 1,
    ) -> tuple[dict[str, dict[str, Any]], str | None, dict[str, str]]:
        """Run one frame; returns ({node_id: {"image":…,"luminance":…}}, terminal_id, node_errors)."""
        raw_nodes = {n["id"]: n for n in nodes}
        gnodes = [GraphNode(**{k: v for k, v in n.items() if k in GraphNode.__dataclass_fields__}) for n in nodes]
        gedges = [GraphEdge(**{k: v for k, v in e.items() if k in GraphEdge.__dataclass_fields__}) for e in edges]

        node_map = {n.id: n for n in gnodes}
        order = self._topo_sort(gnodes, gedges)
        terminal_id = self._find_terminal(gnodes, gedges, order)
        # Stop execution at the terminal — don't run downstream nodes
        if terminal_id and terminal_id in order:
            order = order[: order.index(terminal_id) + 1]

        # ── Build global Timeline for this frame ──────────────────────
        # Check for a Timeline node in the graph — its params override defaults
        tl_node = next((n for n in gnodes if n.method_id == "__timeline__"), None)
        tl_params = tl_node.params if tl_node else {}
        tl_total = int(tl_params.get("total_frames", frames))
        tl_fps = int(tl_params.get("fps", self._fps))
        tl_speed = float(tl_params.get("speed", 1.0))

        # Also check per-node anim_speed — first node with anim_speed wins for global timeline
        for n in gnodes:
            ns = float(n.params.get("anim_speed", 1.0))
            if ns != 1.0:
                tl_speed = ns
                break

        timeline = make_timeline(
            global_frame=frame,
            total_frames=tl_total,
            fps=tl_fps,
            speed=tl_speed,
        )

        flat_outputs: dict[str, dict[str, Any]] = {}
        ran: dict[str, bool] = {}  # node_id → actually executed this frame
        node_errors: dict[str, str] = {}  # node_id → traceback text

        # Diagnostics counters — accumulated per frame
        _diag_node_timings: dict[str, float] = {}   # node_id → ms actually spent in fn()
        _diag_cache_hits:   int = 0
        _diag_cache_misses: int = 0
        _diag_mem_edges:    int = 0  # edges passed as ndarray (no disk write)
        _diag_disk_edges:   int = 0  # edges written to _input.png
        # Per-edge transport, keyed "src->dst": "mem" (in-memory ndarray) or
        # "disk" (_input.png written). Surfaced for the FX overlay so it shows
        # the real transport of each transfer, not a heuristic.
        _diag_edge_transport: dict[str, str] = {}
        _diag_gpu_nodes:    int = 0
        _diag_cpu_nodes:    int = 0
        _diag_nodes_skipped: int = 0
        _diag_nodes_cooked:  int = 0
        _diag_exec_start = time.monotonic()

        for node_id in order:
            node = node_map[node_id]

            # ── Group node: recursive sub-execution ───────────────────
            raw_n = raw_nodes.get(node_id, {})
            if raw_n.get("type") == "group":
                slot, g_errors = self._execute_group_node(
                    raw_n, node_id, flat_outputs, gedges,
                    self.out_dir, seed, frame, frames,
                )
                flat_outputs[node_id] = slot
                node_errors.update(g_errors)
                ran[node_id] = True
                continue

            meta = registry.get_meta(node.method_id)
            if meta is None:
                raise GraphError(f"Unknown method '{node.method_id}'")

            upstream_node_ids = {
                e.src_node for e in gedges if e.dst_node == node_id and not e.feedback
            }
            upstream_ran = any(ran.get(uid, False) for uid in upstream_node_ids)

            # ── Selective recooking (dirty flag) ──────────────────────────────
            if not node.dirty and not upstream_ran:
                # Fast path: in-memory live mode — reuse _prev_outputs if available.
                # This is the main Phase 6 optimisation; disk cache is the fallback
                # for the single-frame / sequence render paths.
                if self._in_memory and node_id in self._prev_outputs:
                    flat_outputs[node_id] = self._prev_outputs[node_id]
                    ran[node_id] = False
                    _diag_nodes_skipped += 1
                    continue

                _cache_dir = self.out_dir / node_id
                if _cache_dir.exists():
                    _pngs_c = sorted(
                        p for p in _cache_dir.glob("*.png") if not p.name.startswith("_")
                    )
                    if _pngs_c:
                        import json as _jc
                        from PIL import Image as _PILc
                        _arr_c = np.array(
                            _PILc.open(str(_pngs_c[-1])).convert("RGB"), dtype=np.float32
                        ) / 255.0
                        _sc_c = (
                            _jc.loads((_cache_dir / "scalars.json").read_text())
                            if (_cache_dir / "scalars.json").exists() else {}
                        )
                        _field_c = (
                            np.load(str(_cache_dir / "field.npy"))
                            if (_cache_dir / "field.npy").exists() else None
                        )
                        _parts_c = (
                            np.load(str(_cache_dir / "particles.npy"))
                            if (_cache_dir / "particles.npy").exists() else None
                        )
                        _mask_c = (
                            np.load(str(_cache_dir / "mask.npy"))
                            if (_cache_dir / "mask.npy").exists() else None
                        )
                        flat_outputs[node_id] = {
                            "image":     _arr_c,
                            "luminance": np.mean(_arr_c, axis=-1) if _arr_c is not None else 0.0,
                            "field":     _field_c if _field_c is not None else _arr_c,
                            "particles": _parts_c,
                            "mask":      _mask_c,
                            **_sc_c,
                        }
                        # Inherit upstream scalars into cached payload
                        _inh = {
                            k: float(v)
                            for uid in upstream_node_ids
                            for k, v in (flat_outputs.get(uid) or {}).items()
                            if k not in ("image", "field", "particles", "mask")
                            and isinstance(v, (int, float))
                        }
                        flat_outputs[node_id] = {**_inh, **flat_outputs[node_id]}
                        ran[node_id] = False
                        print(f"  ↩ {node_id} skipped (clean)")
                        continue

            # ── Architecture A: simulation with capture_frame() ──────
            node_seed = seed + frame + _stable_node_offset(node_id)
            from .arch import detect_architecture
            arch = detect_architecture(meta)
            sim_cache_key = (node_id, seed)
            # Computed unconditionally: the list-result readback below uses it
            # even when the architecture heuristic said "B".
            params_hash = _node_params_hash(node.method_id, node.params)

            if arch == "A":
                # Check if simulation is already cached
                if (sim_cache_key in self._sim_cache
                        and self._sim_params_hash.get(node_id) == params_hash):
                    cached = self._sim_cache[sim_cache_key]
                    if cached:
                        # Loop the cooked frames — never re-cook a cached sim.
                        # The live window (LIVE_TOTAL_FRAMES) can exceed the
                        # cooked frame count; without the modulo, frames past the
                        # end fell through to a full re-cook every frame (~2 fps
                        # after the first few seconds of smooth playback).
                        entry = cached[frame % len(cached)]
                        # List-returning methods cache dicts per frame, the
                        # capture path caches bare ndarrays — handle both.
                        if isinstance(entry, dict):
                            arr = entry.get("image")
                            extra = {k: v for k, v in entry.items()
                                     if k not in ("image", "luminance")}
                        else:
                            arr = entry
                            extra = {}
                        flat_outputs[node_id] = {
                            "image": arr,
                            "luminance": np.mean(arr, axis=-1) if arr is not None else 0.0,
                            "field": extra.get("field", arr),
                            "particles": extra.get("particles"),
                            "mask": extra.get("mask"),
                            **{k: v for k, v in extra.items()
                               if k not in ("field", "particles", "mask")},
                        }
                        ran[node_id] = True
                        _diag_cache_hits += 1
                        continue

                # Need to (re)run the simulation
                # Override n_frames to match the requested frame range
                run_params_preview = dict(node.params)
                if "n_frames" in run_params_preview and frames > 1:
                    run_params_preview["n_frames"] = frames

                # Install capture context
                from image_pipeline.core.animation import (
                    set_job_context, clear_job_context, get_frames, JobCancelled
                )
                import threading as _thr
                _captured = []
                _cancel_evt = _thr.Event()
                def _on_capture(arr):
                    _captured.append(
                        (arr.copy() / 255.0).astype(np.float32)
                        if isinstance(arr, np.ndarray) and arr.dtype == np.uint8
                        else (arr.copy() if isinstance(arr, np.ndarray) else np.array(arr))
                    )

                node_dir_sim = self.out_dir / node_id
                node_dir_sim.mkdir(parents=True, exist_ok=True)

                # Build run params for the full simulation
                sim_params = dict(node.params)
                if "n_frames" in sim_params and frames > 1:
                    sim_params["n_frames"] = frames
                sim_params["_timeline"] = timeline
                sim_params["time"] = timeline.phase
                sim_params["frame"] = 0
                sim_params["frame_seed"] = node_seed

                set_job_context(on_frame=_on_capture, cancel_event=_cancel_evt)
                _t0_arch_a = time.monotonic()
                try:
                    try:
                        from image_pipeline.core import utils as _core_utils
                        _core_utils.set_method_id(meta.id)
                        meta.fn(node_dir_sim, node_seed, params=sim_params)
                    except TypeError as _te:
                        if "unexpected keyword argument" not in str(_te):
                            raise
                        meta.fn(node_dir_sim, node_seed)
                except JobCancelled:
                    pass
                finally:
                    from image_pipeline.core import utils as _core_utils
                    _core_utils.set_method_id(None)
                    clear_job_context()
                _diag_node_timings[node_id] = (time.monotonic() - _t0_arch_a) * 1000.0
                _diag_cache_misses += 1

                # Collect captured frames
                sim_frames = get_frames(meta.id) or _captured
                if not sim_frames:
                    # Fallback: read PNG from disk
                    _pngs = sorted(
                        p for p in node_dir_sim.glob("*.png")
                        if not p.name.startswith("_")
                    )
                    if _pngs:
                        from PIL import Image as _PILfb
                        _arr = np.array(
                            _PILfb.open(str(_pngs[-1])).convert("RGB"),
                            dtype=np.float32
                        ) / 255.0
                        sim_frames = [_arr]

                if sim_frames:
                    self._sim_cache[sim_cache_key] = sim_frames
                    self._sim_params_hash[node_id] = params_hash
                    self._evict_sim_cache()

                    # Loop the cooked frames (matches the cache-hit path above).
                    arr = sim_frames[frame % len(sim_frames)]
                    flat_outputs[node_id] = {
                        "image": arr,
                        "luminance": np.mean(arr, axis=-1),
                        "field": arr,
                        "particles": None,
                        "mask": None,
                    }
                    ran[node_id] = True
                    continue
                # If no frames captured, fall through to normal execution

            run_params = dict(node.params)
            # ── Prebake: run sim ahead before first output frame ──────
            # Multiply n_frames by prebake so Architecture A methods run
            # more internal steps before the first captured frame.
            if node.prebake > 0 and "n_frames" in run_params:
                run_params["n_frames"] = int(run_params["n_frames"]) + node.prebake

            # ── Per-node timeline with timing offset ──────────────────
            # When a node has non-zero end_frame, create a per-node timeline
            # that remaps t/phase to the node's window [start_frame, end_frame).
            # Outside the window: hold at boundary (t=0 before, t=1 after).
            if node.end_frame > 0:
                node_tl = make_timeline(
                    global_frame=timeline.global_frame,
                    total_frames=timeline.total_frames,
                    fps=timeline.fps,
                    speed=timeline.speed,
                    start_frame=node.start_frame,
                    end_frame=node.end_frame,
                )
                run_params["_timeline"] = node_tl
                run_params["time"] = node_tl.phase
            else:
                run_params["_timeline"] = timeline
                # Don't overwrite time if the live loop already injected
                # the raw frame number for continuous evolution.
                if "time" not in run_params:
                    run_params["time"] = timeline.phase
            # Single animation driver: expose the global speed so methods that
            # read anim_speed (the legacy param) stay in sync with the Timeline
            # clock instead of defaulting to 1.0. Guarded so an explicit
            # per-node anim_speed param (or a keyframe) is not clobbered.
            if "anim_speed" not in run_params:
                run_params["anim_speed"] = timeline.speed

            # ── Per-param keyframe evaluation ────────────────────────────
            # Each param has its own independent keyframe track.
            # Evaluate each param's track at the current frame and merge
            # interpolated values into run_params.
            # Keyframe values override node.params but are overridden by
            # explicit wire connections below.
            if node.paramKeyframes:
                for pname, kfs in node.paramKeyframes.items():
                    if not kfs or len(kfs) < 1:
                        continue
                    # Sort by frame
                    sorted_kfs = sorted(kfs, key=lambda k: k.get("frame", 0))
                    val = _evaluate_param_track(sorted_kfs, timeline.global_frame)
                    if val is not None:
                        run_params[pname] = val

            # ── Implicit scalar inheritance (upstream attrs flow without explicit wires) ─
            upstream_scalars: dict[str, float] = {}
            for _uid in upstream_node_ids:
                for _k, _v in (flat_outputs.get(_uid) or {}).items():
                    if (
                        _k not in ("image", "field", "particles", "mask")
                        and isinstance(_v, (int, float))
                        and _k not in upstream_scalars
                    ):
                        upstream_scalars[_k] = float(_v)

            _eligible_s = [k for k, _ in _eligible_params(meta.params or {}, "scalar")]
            for _sk, _sv in upstream_scalars.items():
                _tgt = _score_param(_sk, _eligible_s)
                if _tgt:
                    _inject_typed(run_params, _tgt, _sv, "scalar", node.params)
            # (explicit edges below will override any pre-seeded values)

            node_dir = self.out_dir / node_id
            node_dir.mkdir(parents=True, exist_ok=True)

            image_candidates: list[np.ndarray] = []
            image_edge_keys: list[str] = []  # src->dst of the image edges into this node

            for edge in gedges:
                if edge.dst_node != node_id:
                    continue
                # Non-image (scalar/field/mask/particles) wires are always
                # in-memory value passes — record them as "mem" up front.
                if edge.src_port not in ("image", "field"):
                    _diag_edge_transport.setdefault(f"{edge.src_node}->{edge.dst_node}", "mem")

                # Fetch upstream slot (feedback → previous frame, else current)
                if edge.feedback:
                    _s = self._prev_outputs.get(edge.src_node)
                    slot = _s if _s is not None else {}
                    src_img = slot.get("image")
                    src_lum_raw = slot.get("luminance")
                    src_lum = float(np.mean(src_lum_raw)) if isinstance(src_lum_raw, np.ndarray) else float(src_lum_raw or 0.0)
                    # Black-image fallback so feedback edges work on frame 0
                    if src_img is None and edge.dst_port == "image_in":
                        src_img = np.zeros((H, W, 3), dtype=np.float32)
                else:
                    _s = flat_outputs.get(edge.src_node)
                    slot = _s if _s is not None else {}
                    src_img = slot.get("image")
                    src_lum_raw = slot.get("luminance")
                    src_lum = float(np.mean(src_lum_raw)) if isinstance(src_lum_raw, np.ndarray) else float(src_lum_raw or 0.0)

                # ── IMAGE passthrough ──────────────────────────────────
                if edge.dst_port == "image_in":
                    if src_img is not None:
                        image_candidates.append(src_img)
                        image_edge_keys.append(f"{edge.src_node}->{edge.dst_node}")
                    continue

                # ── Named IMAGE port (e.g. seed_image, mask_image) ─────
                # Port type comes from the method's own declared inputs — no
                # need to rebuild every node def per edge (was O(edges×methods)).
                _port_type = None
                if meta.inputs and edge.dst_port in meta.inputs:
                    _port_type = meta.inputs[edge.dst_port]
                if _port_type and _port_type.lower() == "image" and edge.dst_port != "image_in":
                    if src_img is not None:
                        run_params[edge.dst_port] = src_img
                    else:
                        pass  # no upstream image — leave as None
                    continue

                # ── Named merge-port injection ─────────────────────────
                # The ndarray is always passed in-memory via run_params
                # (the compositing methods prefer it); the temp file is only
                # written in audit mode as a legacy/back-compat fallback.
                if edge.dst_port in ("image_a", "image_b"):
                    if src_img is not None:
                        run_params[edge.dst_port] = src_img
                        _mp_key = f"{edge.src_node}->{edge.dst_node}"
                        if self.audit_to_disk:
                            _p = node_dir / f"_{edge.dst_port}.png"
                            from PIL import Image as _PILmi
                            _PILmi.fromarray(
                                (np.clip(src_img, 0.0, 1.0) * 255).astype(np.uint8)
                            ).save(str(_p))
                            run_params[f"{edge.dst_port}_path"] = str(_p)
                            _diag_edge_transport[_mp_key] = "disk"
                            _diag_disk_edges += 1
                        else:
                            _diag_edge_transport[_mp_key] = "mem"
                            _diag_mem_edges += 1
                    continue

                if edge.dst_port in ("field_a", "field_b"):
                    _farr = slot.get("field")
                    if _farr is None:
                        _farr = src_img
                    if _farr is not None:
                        run_params[edge.dst_port] = _farr
                        if self.audit_to_disk:
                            _p = node_dir / f"_{edge.dst_port}.npy"
                            np.save(str(_p), np.asarray(_farr, dtype=np.float32))
                            run_params[f"{edge.dst_port}_path"] = str(_p)
                    continue

                if edge.dst_port in ("particles_a", "particles_b"):
                    _parr = slot.get("particles")
                    if _parr is not None:
                        run_params[edge.dst_port] = _parr
                        if self.audit_to_disk:
                            _p = node_dir / f"_{edge.dst_port}.npy"
                            np.save(str(_p), np.asarray(_parr, dtype=np.float32))
                            run_params[f"{edge.dst_port}_path"] = str(_p)
                    continue

                # ── PARTICLES wire ────────────────────────────────────
                if edge.src_port == "particles":
                    particles_val = slot.get("particles")
                    if particles_val is not None:
                        run_params[edge.dst_port] = particles_val
                    continue

                # ── MASK wire ─────────────────────────────────────────
                if edge.src_port == "mask":
                    mask_val = slot.get("mask")
                    if mask_val is not None:
                        run_params[edge.dst_port] = mask_val
                    continue

                # ── COLORMAP wire ──────────────────────────────────────
                if edge.src_port == "palette":
                    cm_val = slot.get("palette")
                    if cm_val is not None:
                        run_params[edge.dst_port] = cm_val
                    continue

                # ── Determine source value and wire type ──────────────
                # Check flat_outputs directly so named scalar sidecars (r, amplitude, …)
                # are resolved by port name, not just by hardcoded "luminance".
                slot_val = slot.get(edge.src_port)

                if edge.src_port == "luminance" and isinstance(slot_val, np.ndarray):
                    # luminance is now a per-pixel FIELD (H,W) float32
                    src_val = slot_val
                    src_type = "field"
                elif edge.src_port == "luminance" or (
                    slot_val is not None and isinstance(slot_val, (int, float))
                ):
                    # Named scalar output — could be "luminance" or a sidecar key like "r"
                    raw = slot_val if slot_val is not None else src_lum
                    src_val: Any = (
                        float(np.mean(raw)) if isinstance(raw, np.ndarray) else float(raw)
                    )
                    src_type = "scalar"
                elif edge.src_port in ("image", "field"):
                    # field slot now holds the real field array if written, else image fallback
                    src_val = slot.get(edge.src_port, src_img)
                    src_type = "field"
                else:
                    continue

                # ── Route to target param ─────────────────────────────
                if edge.dst_port in node.params:
                    # User wired to a specific named param port — inject directly,
                    # enforcing type compatibility (logs warning on mismatch).
                    _inject_typed(run_params, edge.dst_port, src_val, src_type, node.params)
                else:
                    # dst_port is not a named param (generic or unrecognised) —
                    # pick the best eligible param by name-similarity scoring.
                    eligible = _eligible_params(meta.params or {}, src_type)
                    target = _score_param(edge.src_port, [k for k, _ in eligible])
                    if target:
                        _inject_typed(run_params, target, src_val, src_type, node.params)
                    elif eligible:
                        # No name match — fall back to first eligible param
                        _inject_typed(run_params, eligible[0][0], src_val, src_type, node.params)

            # Save upstream image to a file so methods can read it via load_input()
            upstream_arr: np.ndarray | None = None
            if image_candidates:
                if len(image_candidates) == 1:
                    upstream_arr = image_candidates[0].astype(np.float32)
                else:
                    # Screen blend: 1 - (1-a)*(1-b) — preserves brightness
                    upstream_arr = image_candidates[0].astype(np.float32).copy()
                    for _cand in image_candidates[1:]:
                        upstream_arr = 1 - (1 - upstream_arr) * (1 - _cand.astype(np.float32))

            if upstream_arr is not None:
                # Inject in-memory array for new-contract methods
                run_params["_input_image"] = upstream_arr
                # Legacy methods (new_image_contract=False) call load_input(params["input_image"])
                # so they need the upstream image written to disk. New-contract methods skip this
                # entirely when running in_memory (live loop), eliminating the per-edge PNG write.
                if not (self._in_memory and meta.new_image_contract):
                    upstream_path = node_dir / "_input.png"
                    from PIL import Image as _PILpre2
                    # compress_level=1: this is a transient transport file for
                    # legacy load_input() readers, not the audit trail — fast
                    # encode beats small size on the hot path (~10× faster).
                    _PILpre2.fromarray((upstream_arr * 255).astype(np.uint8)).save(
                        str(upstream_path), compress_level=1)
                    run_params["input_image"] = str(upstream_path)
                    _diag_disk_edges += 1
                    _edge_kind = "disk"
                else:
                    _diag_mem_edges += 1
                    _edge_kind = "mem"
                # Record the real transport for each image edge into this node.
                for _ek in image_edge_keys:
                    _diag_edge_transport[_ek] = _edge_kind
                # Methods whose source param has an "input_image" choice need it selected explicitly
                src_spec = (meta.params or {}).get("source", {})
                if isinstance(src_spec, dict) and "input_image" in (src_spec.get("choices") or []):
                    run_params["source"] = "input_image"
            else:
                # No upstream image — inject None so methods can check
                run_params["_input_image"] = None

            node_seed = seed + frame + _stable_node_offset(node_id)
            run_params["frame"] = frame
            run_params["frame_seed"] = node_seed

            # Evaluate expression strings in numeric params
            for _pk, _spec in (meta.params or {}).items():
                if _pk not in run_params:
                    continue
                _def = _spec.get("default") if isinstance(_spec, dict) else None
                if not isinstance(_def, (int, float)) or isinstance(_def, bool):
                    continue
                _evaled = eval_param(run_params[_pk], frame, node_seed, frames)
                if _evaled is not run_params[_pk]:
                    run_params[_pk] = _evaled

            # ── In-memory output capture ────────────────────────────
            # save() checks a per-thread sink internally (utils.set_save_capture),
            # so capture works for every method regardless of import style. The
            # old module-attribute monkeypatch missed `from utils import save`
            # callers — most methods — so live frames silently paid a full PNG
            # encode + write + decode read-back per node. It was also process-
            # global (BUG-5); the sink is per-thread, so concurrent jobs can't
            # capture each other's frames.
            _node_capture: dict = {}
            _captured_sidecars = {}
            from image_pipeline.core import utils as _utils_mod
            if self._in_memory:
                _utils_mod.set_save_capture(
                    _node_capture, skip_disk=not self.audit_to_disk
                )
                # Collect sidecars (field/particles/mask/scalars) in memory too,
                # so live mode does zero disk writes — see utils.set_sidecar_context.
                _utils_mod.set_sidecar_context(_captured_sidecars)

            # ── Call the method ──
            _diag_nodes_cooked += 1
            _fn_result = None
            _is_gpu = "gpu" in (meta.tags or [])
            if _is_gpu:
                _diag_gpu_nodes += 1
            else:
                _diag_cpu_nodes += 1
            _t0_node = time.monotonic()
            try:
                from image_pipeline.core import utils as _core_utils
                _core_utils.set_method_id(meta.id)
                _fn_result = meta.fn(node_dir, node_seed, params=run_params)
            except TypeError as _te:
                if "unexpected keyword argument" not in str(_te):
                    raise
                _fn_result = meta.fn(node_dir, node_seed)
            except Exception as exc:
                from image_pipeline.core import utils as _core_utils
                _core_utils.set_method_id(None)
                err_text = traceback.format_exc(limit=8)
                err_img = _write_error_placeholder(node_dir, write=self.audit_to_disk)
                node_errors[node_id] = err_text
                print(f"[node-error] {node_id}: {exc}")
                flat_outputs[node_id] = {
                    "image":     err_img,
                    "luminance": 0.0,
                    "field":     err_img,
                    "particles": None,
                    "mask":      None,
                }
                ran[node_id] = True
                _diag_node_timings[node_id] = (time.monotonic() - _t0_node) * 1000.0
                _diag_cache_misses += 1
                if self._in_memory:
                    _utils_mod.set_save_capture(None)
                    _utils_mod.set_sidecar_context(None)
                continue
            _diag_node_timings[node_id] = (time.monotonic() - _t0_node) * 1000.0
            _diag_cache_misses += 1

            # ── Read back output ────────────────────────────────────
            arr = None
            extra_outputs: dict = {}

            if isinstance(_fn_result, list):
                # Architecture A: list of dicts — cache all frames
                self._sim_cache[sim_cache_key] = _fn_result
                self._sim_params_hash[node_id] = params_hash
                if frame < len(_fn_result):
                    frame_data = _fn_result[frame]
                    arr = frame_data.get("image")
                    extra_outputs = {k: v for k, v in frame_data.items() if k not in ("image", "luminance")}
            elif isinstance(_fn_result, dict):
                # Architecture B: single dict
                arr = _fn_result.get("image")
                extra_outputs = {k: v for k, v in _fn_result.items() if k not in ("image", "luminance")}
            elif isinstance(_fn_result, np.ndarray):
                # Legacy: ndarray → treat as image
                arr = _fn_result
            elif hasattr(_fn_result, 'mode') and hasattr(_fn_result, 'size'):
                # Legacy: PIL Image → treat as image
                arr = np.array(_fn_result, dtype=np.float32) / 255.0
            else:
                # Legacy: None → in-memory capture, then disk read-back fallback
                arr = _node_capture.get("image")
                if arr is None:
                    pngs = sorted(p for p in node_dir.glob("*.png") if not p.name.startswith("_"))
                    if pngs:
                        from PIL import Image
                        img = Image.open(str(pngs[-1])).convert("RGB")
                        arr = np.array(img, dtype=np.float32) / 255.0

            # ── Read sidecar files — for every return type ──
            # Methods may combine a return value with write_scalars /
            # write_field / write_particles sidecars; in-memory values from
            # the return dict take priority over the files. In live (in_memory)
            # mode the sidecar writers already routed into _captured_sidecars,
            # so we read from there first and never touch disk.
            import json as _json
            for _k, _v in (list(_captured_sidecars.items()) if self._in_memory else []):
                extra_outputs.setdefault(_k, _v)
            scalars_path = node_dir / "scalars.json"
            if scalars_path.exists() and self._in_memory is False:
                for _k, _v in _json.loads(scalars_path.read_text()).items():
                    extra_outputs.setdefault(_k, _v)
            for _key in ("field", "particles", "mask"):
                if self._in_memory:
                    continue
                _path = node_dir / f"{_key}.npy"
                if _path.exists() and _key not in extra_outputs:
                    extra_outputs[_key] = np.load(str(_path))

            # ── Write to disk (audit trail / timeline playback) ──
            # Skipped entirely in live mode (audit_to_disk=False). In render
            # mode, skipped when the method already wrote the same file via
            # its own save() call (the capture sink tells us it fired), and —
            # as before — for new-contract methods running in-memory, whose
            # output lives on the payload bus.
            _saved_by_method = self._in_memory and _node_capture.get("image") is not None \
                and self.audit_to_disk
            _mem_new_contract = self._in_memory and meta.new_image_contract
            if arr is not None and self.audit_to_disk and not _saved_by_method \
                    and not _mem_new_contract:
                from PIL import Image as _PIL_write
                arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr
                _PIL_write.fromarray(arr_u8).save(str(node_dir / f"{meta.filename()}"))

            # ── Write sidecar files ──
            # In live (in_memory) mode the writers already routed into memory,
            # so skip disk entirely — matches the image write above.
            if self._in_memory:
                pass
            else:
                for _key in ("field", "particles", "mask"):
                    _val = extra_outputs.get(_key)
                    if _val is not None:
                        np.save(str(node_dir / f"{_key}.npy"), np.asarray(_val, dtype=np.float32))
                _scalars = {k: v for k, v in extra_outputs.items()
                            if isinstance(v, (int, float)) and k not in ("field", "particles", "mask")}
                if _scalars:
                    import json
                    (node_dir / "scalars.json").write_text(json.dumps(_scalars))

            if self._in_memory:
                _utils_mod.set_save_capture(None)
                _utils_mod.set_sidecar_context(None)
            from image_pipeline.core import utils as _core_utils
            _core_utils.set_method_id(None)

            # ── Build flat_outputs ──
            # luminance is always computed as per-pixel grayscale (H,W) float32
            _lum = np.mean(arr, axis=-1) if arr is not None else 0.0
            flat_outputs[node_id] = {
                "image":     arr,
                "luminance": _lum,
                "field":     extra_outputs.get("field", arr),
                "particles": extra_outputs.get("particles"),
                "mask":      extra_outputs.get("mask"),
                **{k: v for k, v in extra_outputs.items()
                   if k not in ("field", "particles", "mask", "luminance")},
            }
            ran[node_id] = True

            # ── Payload inheritance: merge upstream scalars not produced by this node ──
            _inherited = {
                k: float(v)
                for uid in upstream_node_ids
                for k, v in (flat_outputs.get(uid) or {}).items()
                if k not in ("image", "field", "particles", "mask")
                and isinstance(v, (int, float))
            }
            flat_outputs[node_id] = {**_inherited, **flat_outputs[node_id]}

            ran[node_id] = True

        self._prev_outputs = flat_outputs

        # ── Write diagnostics for this frame ──────────────────────────
        _total_node_ms = sum(_diag_node_timings.values())
        _total_exec_ms = (time.monotonic() - _diag_exec_start) * 1000.0
        self.last_frame_stats = {
            "node_timings":   _diag_node_timings,
            "cache_hits":     _diag_cache_hits,
            "cache_misses":   _diag_cache_misses,
            "mem_edges":      _diag_mem_edges,
            "disk_edges":     _diag_disk_edges,
            "edge_transport": _diag_edge_transport,  # {"src->dst": "mem"|"disk"}
            "gpu_nodes":      _diag_gpu_nodes,
            "cpu_nodes":      _diag_cpu_nodes,
            "nodes_cooked":   _diag_nodes_cooked,
            "nodes_skipped":  _diag_nodes_skipped,
            "node_compute_ms": round(_total_node_ms, 2),
            "overhead_ms":    round(max(0.0, _total_exec_ms - _total_node_ms), 2),
        }

        return flat_outputs, terminal_id, node_errors

    def _execute_group_node(
        self,
        group_raw: dict,
        group_id: str,
        flat_outputs: dict[str, dict[str, Any]],
        gedges: list[GraphEdge],
        out_dir: Path,
        seed: int,
        frame: int,
        frames: int,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Execute the subgraph of a group node; return its output slot and any errors."""
        subgraph       = group_raw.get("subgraph", {})
        inner_nodes    = [dict(n) for n in subgraph.get("nodes", [])]
        inner_edges    = list(subgraph.get("edges", []))
        exposed_inputs = group_raw.get("exposed_inputs", [])
        exposed_outputs = group_raw.get("exposed_outputs", [])

        group_dir = out_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)

        # Deep-copy params so we can inject without mutating originals
        inner_nodes = [dict(n, params=dict(n.get("params") or {})) for n in inner_nodes]
        inner_map   = {n["id"]: n for n in inner_nodes}

        # Inject outer wire values into exposed inner params
        for exp_in in exposed_inputs:
            outer_port   = exp_in.get("port", "")
            inner_nid    = exp_in.get("inner_node", "")
            inner_param  = exp_in.get("inner_param", "input_image")

            src_slot: dict[str, Any] = {}
            for edge in gedges:
                if edge.dst_node == group_id and edge.dst_port == outer_port:
                    src_slot = flat_outputs.get(edge.src_node) or {}
                    break

            inner_node = inner_map.get(inner_nid)
            if inner_node is None or not src_slot:
                continue

            if inner_param in ("input_image",):
                img_arr = src_slot.get("image")
                if img_arr is not None:
                    from PIL import Image as _PILinj
                    inp_path = group_dir / f"_inject_{inner_nid}.png"
                    # Transient transport file — fast encode, not audit.
                    _PILinj.fromarray(
                        (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
                    ).save(str(inp_path), compress_level=1)
                    inner_node["params"]["input_image"] = str(inp_path)
            else:
                val = src_slot.get(inner_param, src_slot.get("luminance", 0.0))
                if isinstance(val, (int, float)):
                    inner_node["params"][inner_param] = val

        # Reuse the sub-executor across frames — a fresh one per frame loses
        # feedback state and re-runs Arch-A sims inside the group from
        # scratch every frame (BUG-6). Inherits live-mode transport flags.
        sub_executor = self._group_executors.get(group_id)
        if sub_executor is None:
            sub_executor = GraphExecutor(
                group_dir, fps=self._fps,
                in_memory=self._in_memory, audit_to_disk=self.audit_to_disk,
            )
            self._group_executors[group_id] = sub_executor
        sub_outputs, terminal_id, sub_errors = sub_executor.execute(
            inner_nodes, inner_edges, seed, frame=frame, frames=frames
        )

        # Return output from first exposed_output, else auto-detected terminal
        for exp_out in exposed_outputs:
            inner_src = exp_out.get("inner_node", "")
            if inner_src in sub_outputs:
                return sub_outputs[inner_src], sub_errors

        if terminal_id and terminal_id in sub_outputs:
            return sub_outputs[terminal_id], sub_errors

        return {}, sub_errors

    def selective_invalidate(
        self,
        old_nodes: list[dict],
        new_nodes: list[dict],
        old_edges: list[dict],
        new_edges: list[dict],
        seed: int,
    ) -> int:
        """Invalidate only the sim-cache entries that must be re-cooked after a hot-swap.

        Call this on the persistent executor before re-using it with an updated
        graph. Returns the number of cache entries cleared.

        Rules:
        - Topology change (edge set changed) → flush everything.
        - Node removed → remove its cache entry.
        - Node's non-volatile params changed → remove its cache entry.
        - Node's params unchanged (only volatile keys like 'time' differ) → keep cache.
        """
        def _edge_sig(e: dict) -> tuple:
            return (e.get("src_node", ""), e.get("src_port", ""),
                    e.get("dst_node", ""), e.get("dst_port", ""))

        old_topo = sorted(_edge_sig(e) for e in old_edges)
        new_topo = sorted(_edge_sig(e) for e in new_edges)

        if old_topo != new_topo:
            n = len(self._sim_cache)
            self._sim_cache.clear()
            self._sim_params_hash.clear()
            return n

        old_map = {n["id"]: n for n in old_nodes}
        new_map = {n["id"]: n for n in new_nodes}
        invalidated = 0

        for nid in list(old_map):
            if nid not in new_map:
                if self._sim_cache.pop((nid, seed), None) is not None:
                    invalidated += 1
                self._sim_params_hash.pop(nid, None)

        for nid, node in new_map.items():
            new_hash = _node_params_hash(node.get("method_id", ""), node.get("params", {}))
            old_hash = self._sim_params_hash.get(nid)
            if old_hash is not None and old_hash != new_hash:
                if self._sim_cache.pop((nid, seed), None) is not None:
                    invalidated += 1
                self._sim_params_hash.pop(nid, None)

        return invalidated

    def _find_terminal(
        self, nodes: list[GraphNode], edges: list[GraphEdge], order: list[str]
    ) -> str | None:
        """Return the last render-flagged node in topo order; otherwise the last node with no outgoing non-feedback edges."""
        render_nodes = [n.id for n in nodes if n.render and n.id in order]
        if render_nodes:
            # Return the LAST render-flagged node in topological order
            for nid in reversed(order):
                if nid in render_nodes:
                    return nid
        has_outgoing = {e.src_node for e in edges if not e.feedback}
        node_map = {n.id: n for n in nodes}

        def _produces_image(nid: str) -> bool:
            n = node_map.get(nid)
            if n is None or not n.method_id:
                return True  # group node — assume image-capable
            meta = registry.get_meta(n.method_id)
            if meta is None:
                return True
            return "image" in (meta.outputs or {})

        # Prefer image-producing sinks — data-only nodes (Timeline, LFO,
        # Math, …) are often dangling and must not be picked as terminal.
        for nid in reversed(order):
            if nid not in has_outgoing and _produces_image(nid):
                return nid
        for nid in reversed(order):
            if nid not in has_outgoing:
                return nid
        return order[-1] if order else None

    # ── Topological sort (Kahn's algorithm) ──────────────────────────

    def _topo_sort(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> list[str]:
        ids = [n.id for n in nodes]
        in_degree: dict[str, int] = {nid: 0 for nid in ids}
        adj: dict[str, list[str]] = {nid: [] for nid in ids}

        for e in edges:
            if e.feedback:
                continue  # feedback edges don't constrain execution order
            if e.src_node not in adj or e.dst_node not in in_degree:
                continue
            adj[e.src_node].append(e.dst_node)
            in_degree[e.dst_node] += 1

        queue = deque(nid for nid in ids if in_degree[nid] == 0)
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for nb in adj[nid]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        non_feedback_edges = [e for e in edges if not e.feedback]
        if len(order) != len(ids):
            # Find the cycle-forming non-feedback edges
            raise GraphError(
                "Graph contains a cycle. Mark back-edges as 'feedback' to enable loops."
            )

        return order
