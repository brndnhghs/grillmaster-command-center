"""Genome → flat numeric feature dict for the taste model (plan §9).

All features derive from the graph alone (no rendering). Returned as a dict
so phase-3 visual/embedding keys can slot in without breaking the dataset —
the model vectorizes over the union of keys it has seen.
"""
from __future__ import annotations

from .generator import GenePool, build_gene_pool
from .config import ShootoutConfig, DEFAULT_CONFIG

_ORIGINS = ("random", "mutation", "crossover", "explorer")


def _topo_depth(nodes: list[dict], edges: list[dict]) -> dict[str, int]:
    depth = {n["id"]: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for e in edges:
            if e.get("feedback"):
                continue
            if e["src_node"] in depth and e["dst_node"] in depth:
                want = depth[e["src_node"]] + 1
                if depth[e["dst_node"]] < want:
                    depth[e["dst_node"]] = want
                    changed = True
        if not changed:
            break
    return depth


def genome_features(genome: dict, pool: GenePool | None = None,
                    cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict[str, float]:
    pool = pool or build_gene_pool(cfg)
    graph = genome["graph"]
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    f: dict[str, float] = {
        "n_nodes": float(len(nodes)),
        "n_edges": float(len(edges)),
        "branching": len(edges) / max(len(nodes), 1),
        "has_feedback": float(any(e.get("feedback") for e in edges)),
    }

    depth = _topo_depth(nodes, edges)
    f["depth"] = float(max(depth.values(), default=0))

    n_drivers = n_combiners = 0
    norm_vals: list[float] = []
    n_extreme = 0
    for n in nodes:
        d = pool.defs.get(n.get("method_id"))
        if d is None:
            continue
        # Node-type histogram — per method_id and per category
        f[f"m_{n['method_id']}"] = f.get(f"m_{n['method_id']}", 0.0) + 1.0
        cat = d.get("category") or "unknown"
        f[f"cat_{cat}"] = f.get(f"cat_{cat}", 0.0) + 1.0

        out_types = set((d.get("outputs") or {}).values())
        if out_types <= {"scalar"}:
            n_drivers += 1
        if any(p.endswith(("_a", "_b")) for p in (d.get("inputs") or {})):
            n_combiners += 1

        # Normalized param positions within their schema range
        schema = d.get("params") or {}
        for k, v in (n.get("params") or {}).items():
            spec = schema.get(k)
            if not isinstance(spec, dict) or not isinstance(v, (int, float)) \
                    or isinstance(v, bool):
                continue
            lo, hi = spec.get("min"), spec.get("max")
            if lo is None or hi is None or hi <= lo:
                continue
            t = (v - lo) / (hi - lo)
            norm_vals.append(min(max(t, 0.0), 1.0))
            if t <= 0.02 or t >= 0.98:
                n_extreme += 1

    f["n_drivers"] = float(n_drivers)
    f["n_combiners"] = float(n_combiners)
    if norm_vals:
        mean = sum(norm_vals) / len(norm_vals)
        var = sum((x - mean) ** 2 for x in norm_vals) / len(norm_vals)
        f["param_mean"] = round(mean, 5)
        f["param_spread"] = round(var ** 0.5, 5)
        f["param_extreme_frac"] = round(n_extreme / len(norm_vals), 5)
    else:
        f["param_mean"] = 0.5
        f["param_spread"] = 0.0
        f["param_extreme_frac"] = 0.0

    origin = genome.get("origin", "random")
    for o in _ORIGINS:
        f[f"origin_{o}"] = float(origin == o)

    return f


# ── Behavior features for quality-diversity (Route 8 / Phase 1C #2) ──
# MAP-Elites-style diversity needs a STABLE behavior-characterization space.
# The generator emits motif tags but they are never populated (0/649 in the
# real corpus), so motif-based diversity maintenance is blind. Instead derive a
# low-dim behavior vector from signals that ARE reliably present: the
# structural graph features computed above (always available, no render
# needed) plus, when a genome has been rendered, the evaluator's liveness bands
# and render cost. Behavior features are persisted on every genome at save time
# (store.save_genome), so the MAP-Elites cell map can be seeded from the on-disk
# corpus without re-rendering anything.

_BEHAVIOR_POOL = None


def _behavior_pool() -> "GenePool":
    global _BEHAVIOR_POOL
    if _BEHAVIOR_POOL is None:
        _BEHAVIOR_POOL = build_gene_pool(DEFAULT_CONFIG)
    return _BEHAVIOR_POOL


def _band(v, edges: tuple[float, ...]) -> float:
    """Map a continuous value to an integer bin index (stable for cell hashing).

    Returns -1.0 when ``v`` is not a number (missing signal).
    """
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return -1.0
    b = 0
    for e in edges:
        if v >= e:
            b += 1
    return float(b)


def behavior_features(genome: dict, pool: "GenePool | None" = None,
                      cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """Low-dim behavior-characterization vector for quality-diversity.

    Starts from the structural graph features (no render needed) and enriches
    with evaluator/render bands when present. The vector is JSON-serializable
    (all floats) and stable across runs for the same genome, so persisted
    copies are safe to band into cells.
    """
    pool = pool or _behavior_pool()
    f = genome_features(genome, pool, cfg)

    liv = genome.get("liveness")
    if isinstance(liv, dict):
        f["spec_peak"] = _band(liv.get("spectral_peak"), (0.3, 0.6, 0.9))
        f["flow_var"] = _band(liv.get("flow_var"), (0.05, 0.2, 0.5))
        f["color_corr"] = _band(liv.get("color_struct_corr"), (0.4, 0.7, 0.9))

    dev = genome.get("deviation")
    if isinstance(dev, dict) and dev.get("kind"):
        # crude stable categorical bin for the deviation-kind string
        f["dev_kind"] = float(abs(hash(str(dev["kind"]))) % 8)

    render = genome.get("render")
    if isinstance(render, dict):
        f["cost_band"] = _band(render.get("wall_s"), (30.0, 100.0, 250.0))

    return f


# Compact, meaningful subspace used to discretize behavior into MAP-Elites cells.
_BEHAVIOR_CELL_KEYS = (
    "n_nodes", "depth", "n_drivers", "param_spread", "has_feedback",
    "spec_peak", "flow_var", "color_corr", "dev_kind", "cost_band",
)


def behavior_cell(features: dict[str, float]) -> str:
    """Discretize a behavior-feature vector into a stable MAP-Elites cell key."""
    parts = []
    for k in _BEHAVIOR_CELL_KEYS:
        v = features.get(k)
        if v is None:
            v = -1.0
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = -1.0
        parts.append(f"{k}={round(v, 1)}")
    return "|".join(parts)
