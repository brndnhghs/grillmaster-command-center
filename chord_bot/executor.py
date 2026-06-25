"""Chord Bot graph executor — walks a node graph left-to-right and produces a harmonic sequence.

Architecture mirrors image_pipeline/core/graph.py:
- Nodes are sorted topologically (Kahn's algorithm on edges; x-position as tiebreaker).
- Each node receives the current HarmonicState and returns an updated one.
- Horizontal nodes advance the beat clock by their duration.
- Vertical nodes (augmenters) modify the state without advancing time.
- Per-param keyframe tracks are evaluated at each node's start-beat position.
"""
from __future__ import annotations

import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .registry import get_meta, ChordMeta
from .types import HarmonicState, SequenceEntry
from .keyframes import evaluate_param_tracks


# ── Graph data model ───────────────────────────────────────────────────────────


@dataclass
class ChordNode:
    """Runtime representation of one node in the chord graph."""

    id:             str
    type:           str           # maps to ChordMeta.id in the registry
    x:              float = 0.0
    y:              float = 0.0
    params:         dict[str, Any] = field(default_factory=dict)
    paramKeyframes: dict[str, list[dict]] = field(default_factory=dict)
    dirty:          bool = True


@dataclass
class ChordEdge:
    """A directed edge between two chord nodes."""

    src_node: str
    src_port: str
    dst_node: str
    dst_port: str


# ── Executor ───────────────────────────────────────────────────────────────────


class ChordGraphError(Exception):
    pass


class ChordExecutor:
    """Execute a chord-graph for the full duration and return a harmonic sequence.

    Unlike the image pipeline's per-frame execution, the chord graph is executed
    once to produce the full progression. Each node emits one HarmonicState; the
    total sequence is the ordered list of states from all horizontal nodes.

    Augmentation pipeline per horizontal node:
        1. Evaluate per-param keyframes at the node's start-beat.
        2. Call the horizontal node fn(state, params) → state.
        3. For each vertical augmenter attached to that node (in dependency order),
           call augmenter_fn(state, params) → state.
        4. Record SequenceEntry(state, start_beat, end_beat).
        5. Advance beat clock by state.duration.
    """

    def execute(
        self,
        nodes: list[dict],
        edges: list[dict],
    ) -> list[SequenceEntry]:
        """Run the full graph and return the harmonic sequence.

        Parameters
        ----------
        nodes : list[dict]
            Node dicts with keys: id, type, x, y, params, paramKeyframes.
        edges : list[dict]
            Edge dicts with keys: src_node, src_port, dst_node, dst_port.

        Returns
        -------
        list[SequenceEntry]
            Ordered sequence of harmonic events (one per horizontal node that ran).
        """
        gnodes = [
            ChordNode(
                id=n["id"],
                type=n.get("type", n.get("method_id", "")),
                x=float(n.get("x", 0.0)),
                y=float(n.get("y", 0.0)),
                params=dict(n.get("params", {})),
                paramKeyframes=dict(n.get("paramKeyframes", {})),
                dirty=bool(n.get("dirty", True)),
            )
            for n in nodes
        ]
        gedges = [
            ChordEdge(
                src_node=e["src_node"],
                src_port=e.get("src_port", "harmonic_out"),
                dst_node=e["dst_node"],
                dst_port=e.get("dst_port", "harmonic_in"),
            )
            for e in edges
        ]

        node_map = {n.id: n for n in gnodes}
        order    = self._topo_sort(gnodes, gedges)

        # Classify each node as horizontal or vertical
        node_axis: dict[str, str] = {}
        for nid in order:
            node = node_map[nid]
            meta = get_meta(node.type)
            node_axis[nid] = meta.axis if meta else "horizontal"

        # Map each horizontal node → sorted list of vertical augmenters attached to it
        # An augmenter is attached when its harmonic_in edge comes from a horizontal node.
        aug_map: dict[str, list[str]] = {nid: [] for nid in order if node_axis.get(nid) == "horizontal"}
        for edge in gedges:
            if node_axis.get(edge.dst_node) == "vertical" and node_axis.get(edge.src_node) == "horizontal":
                if edge.src_node in aug_map:
                    aug_map[edge.src_node].append(edge.dst_node)

        # Build a sub-order for augmenters of each horizontal node
        # (sort by topological position within the full order to respect dependencies)
        order_index = {nid: i for i, nid in enumerate(order)}
        for src_nid in aug_map:
            aug_map[src_nid].sort(key=lambda nid: order_index.get(nid, 0))

        # Initialise state with a silent placeholder
        state = HarmonicState()
        beat  = 0.0
        sequence: list[SequenceEntry] = []
        executed: set[str] = set()

        for nid in order:
            node = node_map[nid]
            axis = node_axis.get(nid, "horizontal")

            # Vertical augmenters are driven by their parent horizontal node
            if axis == "vertical":
                continue

            meta = get_meta(node.type)
            if meta is None:
                raise ChordGraphError(f"Unknown chord node type: {node.type!r}")

            # ── Evaluate per-param keyframes at start beat ────────────────────
            run_params = dict(node.params)
            kf_vals = evaluate_param_tracks(node.paramKeyframes, beat)
            run_params.update(kf_vals)

            # ── Execute horizontal node ───────────────────────────────────────
            result = None
            try:
                result = meta.fn(state.copy(), run_params)
            except Exception as exc:
                err = traceback.format_exc(limit=6)
                print(f"[chord-error] {nid} ({node.type}): {exc}\n{err}")
                # Continue with unmodified state so downstream nodes still run

            executed.add(nid)

            # ── Normalise result: single state or phrase list ─────────────────
            # A phrase-type node may return list[HarmonicState]; single nodes
            # return HarmonicState. Both paths apply augmenters and record entries.
            if result is None:
                sub_states: list[HarmonicState] = [state]
            elif isinstance(result, list):
                sub_states = result if result else [state]
            else:
                state      = result
                sub_states = [state]

            # ── Apply vertical augmenters to each sub-state ───────────────────
            aug_nids = aug_map.get(nid, [])
            if aug_nids:
                aug_metas_params: list[tuple] = []
                for aug_nid in aug_nids:
                    aug_node = node_map[aug_nid]
                    aug_meta = get_meta(aug_node.type)
                    if aug_meta is None:
                        continue
                    aug_params = dict(aug_node.params)
                    aug_kf = evaluate_param_tracks(aug_node.paramKeyframes, beat)
                    aug_params.update(aug_kf)
                    aug_metas_params.append((aug_nid, aug_meta, aug_params))
                    executed.add(aug_nid)

                augmented: list[HarmonicState] = []
                for sub in sub_states:
                    s = sub.copy()
                    for aug_nid, aug_meta, aug_params in aug_metas_params:
                        try:
                            s = aug_meta.fn(s.copy(), aug_params)
                        except Exception as exc:
                            print(f"[chord-error] augmenter {aug_nid} ({aug_meta.id}): {exc}")
                    augmented.append(s)
                sub_states = augmented

            # ── Record entries and advance beat clock ─────────────────────────
            for sub in sub_states:
                duration = max(0.0, float(sub.duration))
                entry = SequenceEntry(
                    state=sub.copy(),
                    start_beat=beat,
                    end_beat=beat + duration,
                    node_id=nid,
                )
                sequence.append(entry)
                beat += duration

            # The last sub-state becomes the context for downstream nodes
            state = sub_states[-1]

        return sequence

    # ── Topological sort (Kahn's algorithm, x-position tiebreaker) ────────────

    def _topo_sort(self, nodes: list[ChordNode], edges: list[ChordEdge]) -> list[str]:
        """Kahn's topological sort. x-position breaks ties so left-to-right order holds."""
        ids = [n.id for n in nodes]
        x_pos = {n.id: n.x for n in nodes}

        in_degree: dict[str, int] = {nid: 0 for nid in ids}
        adj: dict[str, list[str]] = {nid: [] for nid in ids}

        for e in edges:
            if e.src_node not in adj or e.dst_node not in in_degree:
                continue
            adj[e.src_node].append(e.dst_node)
            in_degree[e.dst_node] += 1

        # Initial queue: nodes with no predecessors, sorted by x position
        queue: deque[str] = deque(
            sorted((nid for nid in ids if in_degree[nid] == 0), key=lambda n: x_pos[n])
        )
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            # Sort successors by x so tied nodes process left-to-right
            for nb in sorted(adj[nid], key=lambda n: x_pos[n]):
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        if len(order) != len(ids):
            raise ChordGraphError(
                "Graph contains a cycle — mark back-edges or reorder nodes."
            )

        return order
