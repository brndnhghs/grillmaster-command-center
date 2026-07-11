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
