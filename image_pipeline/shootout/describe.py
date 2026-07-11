"""Human-readable clip descriptions for the shootout UI.

Turns a genome graph into:
  * node names (method_id → human name) for every node,
  * a compact graph (ids, names, render flag, typed edges) the UI can draw,
  * a one-line "what this sample is going for" blurb derived from the motif
    provenance, driver coverage, and structure.

No rendering, no LLM — deterministic and cheap, so it can run per clip on
the survivor view.
"""
from __future__ import annotations

from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import GenePool, build_gene_pool

# Friendly names for motif provenance tags (graph["motifs"] entries).
_MOTIF_LABELS = {
    "sim_backbone": "a source-into-filters chain",
    "pattern_blend": "two branches blended together",
    "masked_composite": "one texture masked by another",
    "field_modulate": "a field spatially modulating a node",
    "post_fx": "a post-fx extension of the chain",
    "feedback_loop": "a feedback loop (trails / decay)",
}

# How to describe a driver method by id fragment.
_DRIVER_WORDS = {
    "lfo": "oscillating", "ramp": "ramping", "noise1d": "noise-driven",
    "strobe": "strobing", "envelope": "enveloped", "counter": "stepping",
}


def _name(pool: GenePool, mid: str) -> str:
    return pool.defs.get(mid, {}).get("name", mid)


def node_names(graph: dict, pool: GenePool) -> dict[str, str]:
    """method_id → human name for every node in the graph."""
    out = {}
    for n in graph.get("nodes", []):
        mid = n.get("method_id")
        if mid is not None:
            out[mid] = _name(pool, mid)
    return out


def compact_graph(graph: dict, pool: GenePool) -> dict:
    """Minimal graph the UI can render as a wiring diagram.

    nodes: [{id, method_id, name, render, is_driver}]
    edges: [{src, dst, src_port, dst_port, is_feedback}]
    """
    driver_ids = {n["id"] for n in graph.get("nodes", [])
                  if n.get("method_id") in pool.scalar_drivers}
    nodes = [{
        "id": n["id"],
        "method_id": n.get("method_id"),
        "name": _name(pool, n.get("method_id", "")),
        "render": bool(n.get("render")),
        "is_driver": n["id"] in driver_ids,
    } for n in graph.get("nodes", [])]
    edges = [{
        "src": e.get("src_node"),
        "dst": e.get("dst_node"),
        "src_port": e.get("src_port"),
        "dst_port": e.get("dst_port"),
        "is_feedback": bool(e.get("feedback")),
    } for e in graph.get("edges", [])]
    return {"nodes": nodes, "edges": edges}


def describe_clip(graph: dict, pool: GenePool | None = None,
                  cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict:
    """Blurb + structural facts describing what a sample is going for."""
    pool = pool or build_gene_pool(cfg)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    motifs = graph.get("motifs") or []

    n_nodes = len(nodes)
    driver_nodes = [n for n in nodes
                   if n.get("method_id") in pool.scalar_drivers]
    n_drivers = len(driver_nodes)

    # driver flavor words from the driver method ids
    driver_words = []
    for d in driver_nodes:
        mid = d.get("method_id", "")
        for frag, word in _DRIVER_WORDS.items():
            if frag in mid:
                driver_words.append(word)
                break
    driver_words = list(dict.fromkeys(driver_words))  # dedupe, keep order

    # Build the "going for" blurb.
    if motifs:
        motif_text = ", ".join(str(_MOTIF_LABELS.get(m, m)) for m in motifs[:3])
        blurb = f"Built as {motif_text}."
    else:
        blurb = "A randomly assembled graph."
    if n_drivers:
        if driver_words:
            dw = "/".join(driver_words[:2])
            blurb += f" Animated by {n_drivers} {dw} control node(s)."
        else:
            blurb += f" Animated by {n_drivers} control node(s)."
    else:
        blurb += " Static (no animation drivers attached)."

    return {
        "blurb": blurb,
        "motifs": list(motifs),
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "n_drivers": n_drivers,
        "driver_words": driver_words,
    }
