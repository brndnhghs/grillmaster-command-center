"""Node graph system — wires registered methods into a DAG and executes it."""
from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from . import registry
from .utils import W, H


# ── Port types ────────────────────────────────────────────────────────

class PortType(str, Enum):
    IMAGE     = "image"
    FIELD     = "field"
    PARTICLES = "particles"
    SCALAR    = "scalar"
    ANY       = "any"


# ── Node / edge schema ────────────────────────────────────────────────

@dataclass
class NodeDef:
    method_id:   str
    inputs:      dict[str, str]   # port_name -> PortType value
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
    id:        str
    method_id: str
    params:    dict[str, Any] = field(default_factory=dict)
    x:         float = 0.0
    y:         float = 0.0


# ── Auto-generate NodeDefs from registry ─────────────────────────────

def _make_node_def(meta: registry.MethodMeta) -> NodeDef:
    tags = set(meta.tags)

    # Outputs: always image + luminance scalar; conditionally field / particles
    outputs: dict[str, str] = {
        "image":     PortType.IMAGE,
        "luminance": PortType.SCALAR,
    }
    if tags & {"particle", "agents", "particles"}:
        outputs["particles"] = PortType.PARTICLES
    if tags & {"noise", "field"}:
        outputs["field"] = PortType.FIELD

    # Inputs: image_in (feeds input_image) + one port per wireable param
    inputs: dict[str, str] = {"image_in": PortType.IMAGE}
    param_ports: set[str] = set()

    for pname, spec in (meta.params or {}).items():
        default = spec.get("default") if isinstance(spec, dict) else None
        if default is None:
            continue
        if isinstance(default, bool):
            continue  # bools aren't meaningful as wired values
        if isinstance(default, str):
            continue  # categorical choices aren't wireable
        if isinstance(default, (int, float)):
            inputs[pname] = PortType.SCALAR
            param_ports.add(pname)
        elif isinstance(default, (list, tuple)):
            inputs[pname] = PortType.FIELD
            param_ports.add(pname)

    return NodeDef(method_id=meta.id, inputs=inputs, outputs=outputs, param_ports=param_ports)


def get_all_node_defs() -> dict[str, dict]:
    """Return serialisable NodeDef dict keyed by method_id."""
    result = {}
    for mid, meta in registry.get_all().items():
        nd = _make_node_def(meta)
        result[mid] = {
            "method_id":  nd.method_id,
            "name":       meta.name,
            "category":   meta.category,
            "tags":       meta.tags,
            "params":     meta.params,
            "inputs":     dict(nd.inputs),
            "outputs":    dict(nd.outputs),
            "param_ports": list(nd.param_ports),
        }
    return result


# ── Graph executor ────────────────────────────────────────────────────

class GraphError(Exception):
    pass


class GraphExecutor:
    """Execute a node graph for one or more frames."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        # Keyed by node_id → {"image": ndarray | None, "luminance": float}
        self._prev_outputs: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        nodes: list[dict],
        edges: list[dict],
        seed: int,
        frame: int = 0,
        frames: int = 1,
    ) -> tuple[dict[str, dict[str, Any]], str | None]:
        """Run one frame; returns ({node_id: {"image":…,"luminance":…}}, terminal_id)."""
        gnodes = [GraphNode(**{k: v for k, v in n.items() if k in GraphNode.__dataclass_fields__}) for n in nodes]
        gedges = [GraphEdge(**{k: v for k, v in e.items() if k in GraphEdge.__dataclass_fields__}) for e in edges]

        node_map = {n.id: n for n in gnodes}
        order = self._topo_sort(gnodes, gedges)
        terminal_id = self._find_terminal(gnodes, gedges, order)

        # time ∈ [0, 2π) — full cycle over the animation
        t = (frame / max(1, frames - 1)) * 2 * math.pi if frames > 1 else 0.0

        flat_outputs: dict[str, dict[str, Any]] = {}

        for node_id in order:
            node = node_map[node_id]
            meta = registry.get_meta(node.method_id)
            if meta is None:
                raise GraphError(f"Unknown method '{node.method_id}'")

            run_params = dict(node.params)
            run_params["time"] = t

            image_candidates: list[np.ndarray] = []

            for edge in gedges:
                if edge.dst_node != node_id:
                    continue

                # Fetch upstream slot (feedback → previous frame, else current)
                if edge.feedback:
                    _s = self._prev_outputs.get(edge.src_node)
                    slot = _s if _s is not None else {}
                    src_img = slot.get("image")
                    src_lum = float(slot.get("luminance") or 0.0)
                    # Black-image fallback so feedback edges work on frame 0
                    if src_img is None and edge.dst_port == "image_in":
                        src_img = np.zeros((H, W, 3), dtype=np.float32)
                else:
                    _s = flat_outputs.get(edge.src_node)
                    slot = _s if _s is not None else {}
                    src_img = slot.get("image")
                    src_lum = float(slot.get("luminance") or 0.0)

                if edge.dst_port == "image_in":
                    if src_img is not None:
                        image_candidates.append(src_img)
                elif edge.src_port == "luminance":
                    # Wire a luminance scalar into a numeric param
                    orig = node.params.get(edge.dst_port)
                    # src_lum is float, but guard against any ndarray leakage
                    val = float(np.mean(src_lum)) if isinstance(src_lum, np.ndarray) else float(src_lum)
                    try:
                        run_params[edge.dst_port] = type(orig)(val) if orig is not None else val
                    except (TypeError, ValueError):
                        run_params[edge.dst_port] = val
                elif edge.src_port in ("image", "field"):
                    # If target param is a scalar, reduce the array to its mean
                    # rather than putting a multi-element ndarray into a numeric slot.
                    orig = node.params.get(edge.dst_port)
                    if src_img is not None:
                        if isinstance(orig, (int, float)) and not isinstance(orig, bool):
                            run_params[edge.dst_port] = type(orig)(float(np.mean(src_img)))
                        else:
                            run_params[edge.dst_port] = src_img

            # Save upstream image to a file so methods can read it via load_input()
            upstream_arr: np.ndarray | None = None
            if image_candidates:
                upstream_arr = np.mean(image_candidates, axis=0).astype(np.float32)

            node_dir = self.out_dir / node_id
            node_dir.mkdir(parents=True, exist_ok=True)

            if upstream_arr is not None:
                # Write upstream to disk so methods can load it via load_input(params["input_image"])
                upstream_path = node_dir / "_input.png"
                from PIL import Image as _PILpre2
                _PILpre2.fromarray((upstream_arr * 255).astype(np.uint8)).save(str(upstream_path))
                run_params["input_image"] = str(upstream_path)
                # Methods whose source param has an "input_image" choice need it selected explicitly
                src_spec = (meta.params or {}).get("source", {})
                if isinstance(src_spec, dict) and "input_image" in (src_spec.get("choices") or []):
                    run_params["source"] = "input_image"

            try:
                # Give each node a distinct seed so generator nodes in the same
                # frame don't produce identical outputs. The low 16 bits of
                # hash(node_id) are stable within a process and cheap to compute.
                node_seed = seed + frame + (hash(node_id) & 0xFFFF)
                meta.fn(node_dir, node_seed, params=run_params)
            except TypeError as _e:
                if "unexpected keyword argument" not in str(_e):
                    raise GraphError(f"Node {node_id} ({meta.id}) failed: {_e}") from _e
                try:
                    meta.fn(node_dir, node_seed)
                except Exception as exc:
                    raise GraphError(f"Node {node_id} ({meta.id}) failed: {exc}") from exc
            except Exception as exc:
                raise GraphError(f"Node {node_id} ({meta.id}) failed: {exc}") from exc

            # Read back the produced PNG and compute luminance
            pngs = sorted(node_dir.glob("*.png"))
            arr = None
            if pngs:
                from PIL import Image
                img = Image.open(str(pngs[-1])).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0

                # Filter nodes: meta.fn() already processed upstream via input_image path param.
                # Generator nodes: blend generated output over upstream as a base layer.
                if upstream_arr is not None:
                    is_filter = (meta.category == "filters") or bool(
                        set(meta.tags or []) & {"opencv", "glitch", "dither", "halftone", "distort"}
                    )
                    if not is_filter:
                        from .compositing import blend_two as _blend
                        up = upstream_arr
                        if up.shape != arr.shape:
                            up = np.array(
                                Image.fromarray((up * 255).astype(np.uint8)).resize(
                                    (arr.shape[1], arr.shape[0]), Image.LANCZOS
                                ),
                                dtype=np.float32,
                            ) / 255.0
                        arr = np.clip(_blend(up, arr, "overlay"), 0, 1)
                        Image.fromarray((arr * 255).astype(np.uint8)).save(str(pngs[-1]))

            flat_outputs[node_id] = {
                "image":     arr,
                "luminance": float(np.mean(arr)) if arr is not None else 0.0,
            }

        self._prev_outputs = flat_outputs
        return flat_outputs, terminal_id

    def _find_terminal(
        self, nodes: list[GraphNode], edges: list[GraphEdge], order: list[str]
    ) -> str | None:
        """Last node in topo order with no outgoing non-feedback edges."""
        has_outgoing = {e.src_node for e in edges if not e.feedback}
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
