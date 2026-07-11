"""Per-node contribution analysis — which nodes actually shape the output.

A wild-sampled graph is type-valid but not necessarily *effective*: a branch
can dangle off the output, a driver can animate a param nothing looks at, or a
filter can sit in the chain contributing nothing visible (identity settings, a
zero-weight blend, a fully masked layer). This module diagnoses that.

Two failure modes, cheapest first:

  disconnected — the node's output never reaches the render node. Pure
                 topology, no rendering: a backward reachability walk from
                 the terminal. Definitive.
  silent       — the node IS wired to the output, but ablating it changes
                 nothing visible. Found by re-rendering the graph with the
                 node bypassed/removed and comparing pixels to a baseline.

Ablation strategy per node:

  bypass  — a pass-through filter (exactly one incoming IMAGE edge) is spliced
            out: its consumers are rewired straight to its image source. This
            isolates the node's *own* transformation.
  remove  — a source, a scalar driver, or a multi-input combiner (bypass is
            ill-defined) is deleted along with its edges. A driver revert-to-
            default measures whether it animates anything; a source removal
            measures whether its branch reaches the eye.

Verdicts: terminal (the output node itself), essential (ablation kills a live
clip), contributes (delta above threshold), silent (delta ~0), disconnected.
The union of disconnected+silent is the headline "not contributing" set.

Rendering reuses evaluator.render_stack, so the baseline and every probe share
the exact executor path, n_frames pinning, and liveness downsampling.
"""
from __future__ import annotations

import numpy as np

from . import evaluator
from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import GenePool, build_gene_pool


# ── Topology ──────────────────────────────────────────────────────────


def _terminal_node_id(graph: dict) -> str | None:
    """The node whose image is the clip output: the render-flagged node,
    else a sink (no outgoing edge), else the last node."""
    nodes = graph.get("nodes", [])
    if not nodes:
        return None
    rid = next((n["id"] for n in nodes if n.get("render")), None)
    if rid is not None:
        return rid
    srcs = {e.get("src_node") for e in graph.get("edges", [])}
    sinks = [n["id"] for n in nodes if n["id"] not in srcs]
    return sinks[-1] if sinks else nodes[-1]["id"]


def reachable_from_terminal(graph: dict) -> set[str]:
    """Set of node ids with a directed path to the terminal (i.e. that can
    influence the output). Everything else is disconnected dead weight."""
    term = _terminal_node_id(graph)
    if term is None:
        return set()
    rev: dict[str, list[str]] = {}
    for e in graph.get("edges", []):
        rev.setdefault(e.get("dst_node"), []).append(e.get("src_node"))
    seen = {term}
    stack = [term]
    while stack:
        cur = stack.pop()
        for src in rev.get(cur, []):
            if src is not None and src not in seen:
                seen.add(src)
                stack.append(src)
    return seen


def _input_type(pool: GenePool, method_id: str, port: str) -> str | None:
    return (pool.defs.get(method_id, {}).get("inputs") or {}).get(port)


def ablate(graph: dict, node_id: str, pool: GenePool
           ) -> tuple[list[dict], list[dict], str]:
    """Build a variant graph with `node_id` ablated.

    A single-IMAGE-input filter is bypassed (consumers rewired to its image
    source); anything else is removed with its edges. Returns
    (nodes, edges, mode) — fresh dicts; the input graph is untouched."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    by_id = {n["id"]: n for n in nodes}
    victim = by_id.get(node_id)
    mid = victim.get("method_id") if victim else None

    incoming_img = [
        e for e in edges
        if e.get("dst_node") == node_id
        and _input_type(pool, mid, e.get("dst_port")) == "image"
    ]
    keep_nodes = [dict(n) for n in nodes if n["id"] != node_id]

    if len(incoming_img) == 1:
        src = incoming_img[0]["src_node"]
        src_mid = by_id[src]["method_id"]
        new_edges: list[dict] = []
        for e in edges:
            if e.get("dst_node") == node_id:
                continue  # drop the node's inputs
            if e.get("src_node") == node_id:
                # rewire this consumer to draw straight from the source
                dst_type = _input_type(pool, by_id[e["dst_node"]]["method_id"],
                                       e.get("dst_port")) or "image"
                sp = pool.output_port_for(src_mid, dst_type)
                if sp is None:
                    continue  # source can't feed this port — just sever it
                new_edges.append({**e, "src_node": src, "src_port": sp})
            else:
                new_edges.append(dict(e))
        return keep_nodes, new_edges, "bypass"

    # remove: drop the node and every edge touching it
    new_edges = [dict(e) for e in edges
                 if e.get("src_node") != node_id
                 and e.get("dst_node") != node_id]
    return keep_nodes, new_edges, "remove"


# ── Pixel delta ───────────────────────────────────────────────────────


def _stack(acc: "evaluator.LivenessAccumulator") -> np.ndarray | None:
    """Downsampled grayscale frames as one (T, h, w) array, or None if the
    variant produced no frames. Frames may differ in size (a node can change
    canvas) — crop to the common minimum, matching the liveness path."""
    if not acc.small:
        return None
    h = min(f.shape[0] for f in acc.small)
    w = min(f.shape[1] for f in acc.small)
    return np.stack([f[:h, :w] for f in acc.small])


def _delta(base: np.ndarray | None, other: np.ndarray | None) -> float | None:
    """Mean absolute frame difference, normalized to the [0,1] fraction of
    the combined dynamic range. None if either side produced nothing."""
    if base is None or other is None:
        return None
    t = min(base.shape[0], other.shape[0])
    h = min(base.shape[1], other.shape[1])
    w = min(base.shape[2], other.shape[2])
    b = base[:t, :h, :w]
    o = other[:t, :h, :w]
    scale = float(max(b.max(), o.max()) - min(b.min(), o.min()))
    if scale < 1e-6:  # both effectively constant/black
        return 0.0
    return float(np.abs(b - o).mean() / scale)


# ── Analysis ──────────────────────────────────────────────────────────


def analyze_contribution(genome: dict, cfg: ShootoutConfig = DEFAULT_CONFIG,
                         pool: GenePool | None = None,
                         progress_cb=None) -> dict:
    """Diagnose each node's contribution to the rendered clip.

    Renders a short baseline, then one probe per ablatable node, and
    classifies each. On graphs larger than cfg.contrib_max_nodes the render
    pass is skipped (too many re-renders) and only the free structural
    disconnected-node check runs — `rendered` in the result says which.
    """
    pool = pool or build_gene_pool(cfg)
    graph = genome.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    seed = int(genome.get("seed", 42))
    frames = getattr(cfg, "contrib_frames", 12)
    max_nodes = getattr(cfg, "contrib_max_nodes", 24)
    silent_delta = getattr(cfg, "contrib_silent_delta", 0.01)

    term = _terminal_node_id(graph)
    reach = reachable_from_terminal(graph)
    disconnected = [n["id"] for n in nodes
                    if n["id"] != term and n["id"] not in reach]

    def _rec(n: dict, **extra) -> dict:
        mid = n.get("method_id")
        return {
            "node_id": n["id"],
            "method_id": mid,
            "name": pool.defs.get(mid, {}).get("name", mid),
            "is_driver": mid in pool.scalar_drivers,
            "reachable": n["id"] in reach,
            **extra,
        }

    # Structural-only path for oversized graphs (or empty ones).
    do_render = 0 < len(nodes) <= max_nodes
    if not do_render:
        per_node = []
        for n in nodes:
            if n["id"] == term:
                per_node.append(_rec(n, verdict="terminal", delta=None, mode=None))
            elif n["id"] in disconnected:
                per_node.append(_rec(n, verdict="disconnected", delta=0.0, mode=None))
            else:
                per_node.append(_rec(n, verdict="unprobed", delta=None, mode=None))
        return _report(genome, nodes, term, per_node, disconnected, [],
                       baseline_alive=None, frames=frames, rendered=False)

    if progress_cb:
        progress_cb(f"contribution: baseline render ({frames}f, {len(nodes)} nodes)")
    base_acc = evaluator.render_stack(nodes, edges, seed, cfg, frames, progress_cb)
    base_stack = _stack(base_acc)
    baseline_alive = bool(base_acc.stats().get("alive"))

    per_node: list[dict] = []
    silent: list[str] = []
    for n in nodes:
        nid = n["id"]
        if nid == term:
            per_node.append(_rec(n, verdict="terminal", delta=None, mode=None))
            continue
        if nid in disconnected:
            per_node.append(_rec(n, verdict="disconnected", delta=0.0, mode=None))
            continue

        abl_nodes, abl_edges, mode = ablate(graph, nid, pool)
        acc = evaluator.render_stack(abl_nodes, abl_edges, seed, cfg, frames)
        alive = bool(acc.stats().get("alive"))
        delta = _delta(base_stack, _stack(acc))

        if delta is None or (baseline_alive and not alive):
            verdict = "essential"   # ablation destroyed the output
        elif delta < silent_delta:
            verdict = "silent"
            silent.append(nid)
        else:
            verdict = "contributes"
        rec = _rec(n, verdict=verdict, mode=mode,
                   delta=None if delta is None else round(delta, 5))
        per_node.append(rec)
        if progress_cb:
            progress_cb(f"  {nid} {n.get('method_id')} → {verdict}"
                        f" (Δ={rec['delta']}, {mode})")

    return _report(genome, nodes, term, per_node, disconnected, silent,
                   baseline_alive=baseline_alive, frames=frames, rendered=True)


def _report(genome, nodes, term, per_node, disconnected, silent,
            *, baseline_alive, frames, rendered) -> dict:
    return {
        "genome_id": genome.get("genome_id"),
        "n_nodes": len(nodes),
        "terminal": term,
        "baseline_alive": baseline_alive,
        "rendered": rendered,
        "contrib_frames": frames,
        "per_node": per_node,
        "disconnected": disconnected,          # never reaches the output
        "silent": silent,                      # wired in, no visible effect
        "dead_weight": disconnected + silent,  # the "not contributing" set
    }


def summarize(report: dict) -> str:
    """One-line human summary for logs / progress callbacks."""
    dw = report["dead_weight"]
    head = (f"{report['n_nodes']} nodes · "
            f"{len(report['disconnected'])} disconnected, "
            f"{len(report['silent'])} silent → {len(dw)} not contributing")
    if not report.get("rendered"):
        head += " (structural only — graph too large to ablate)"
    elif report.get("baseline_alive") is False:
        head += " (baseline not alive — silent verdicts unreliable)"
    return head
