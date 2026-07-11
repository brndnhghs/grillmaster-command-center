"""Repair pass — make a genome valid, cheaply, before rendering (plan §7).

Guarantees after repair:
  * every node's method_id exists in the catalog
  * every edge is port-type-legal (src output accepted by dst input/param)
  * the non-feedback edge set is a DAG
  * exactly one node has render:true and it produces IMAGE
  * every node is an ancestor of the terminal (no dead islands)
  * params are clamped/coerced to their schema

Returns None when the graph cannot be made renderable — caller resamples.
"""
from __future__ import annotations

import math
import random

from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import GenePool, build_gene_pool


def _param_wire_type(spec) -> str | None:
    """Wire type of a param used as an implicit input port (mirrors
    core.graph._make_node_def): numeric default → scalar, list/tuple → field."""
    if not isinstance(spec, dict):
        return None
    default = spec.get("default")
    if isinstance(default, bool):
        return None
    if isinstance(default, (int, float)):
        return "scalar"
    if isinstance(default, (list, tuple)):
        return "field"
    return None


def _dst_port_type(pool: GenePool, method_id: str, port: str) -> str | None:
    d = pool.defs.get(method_id)
    if d is None:
        return None
    if port in (d.get("inputs") or {}):
        return d["inputs"][port]
    return _param_wire_type((d.get("params") or {}).get(port))


def _edge_legal(pool: GenePool, nodes_by_id: dict, edge: dict) -> bool:
    src = nodes_by_id.get(edge.get("src_node"))
    dst = nodes_by_id.get(edge.get("dst_node"))
    if src is None or dst is None or src is dst:
        return False
    src_def = pool.defs.get(src["method_id"])
    if src_def is None:
        return False
    src_type = (src_def.get("outputs") or {}).get(edge.get("src_port"))
    if src_type is None:
        return False
    dst_type = _dst_port_type(pool, dst["method_id"], edge.get("dst_port"))
    if dst_type is None:
        return False
    # SCALAR wires into scalar params; FIELD wires into scalar params run
    # through the executor's _field_<param> mechanism — registry rules only.
    return pool.accepts(src_type, dst_type)


def _dagify(nodes: list[dict], edges: list[dict]) -> list[dict]:
    """Keep edges in order, dropping any non-feedback edge that closes a cycle."""
    reach: dict[str, set[str]] = {n["id"]: set() for n in nodes}  # id → ancestors
    kept: list[dict] = []
    for e in edges:
        if e.get("feedback"):
            kept.append(e)
            continue
        src, dst = e["src_node"], e["dst_node"]
        if src == dst or src in _descendants_of(dst, kept):
            continue  # would close a cycle
        kept.append(e)
    return kept


def _descendants_of(nid: str, edges: list[dict]) -> set[str]:
    out: set[str] = set()
    frontier = [nid]
    fwd: dict[str, list[str]] = {}
    for e in edges:
        if not e.get("feedback"):
            fwd.setdefault(e["src_node"], []).append(e["dst_node"])
    while frontier:
        cur = frontier.pop()
        for child in fwd.get(cur, ()):  # noqa: B905
            if child not in out:
                out.add(child)
                frontier.append(child)
    return out


def _ancestors_of(nid: str, edges: list[dict]) -> set[str]:
    out: set[str] = set()
    back: dict[str, list[str]] = {}
    for e in edges:
        if not e.get("feedback"):
            back.setdefault(e["dst_node"], []).append(e["src_node"])
    frontier = [nid]
    while frontier:
        cur = frontier.pop()
        for parent in back.get(cur, ()):
            if parent not in out:
                out.add(parent)
                frontier.append(parent)
    return out


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


def clamp_params(pool: GenePool, node: dict) -> None:
    """Clamp numerics to [min,max], coerce int/bool, reset invalid enums."""
    schema = pool.defs[node["method_id"]].get("params") or {}
    params = node.get("params") or {}
    cleaned: dict = {}
    for k, v in params.items():
        spec = schema.get(k)
        if not isinstance(spec, dict):
            continue  # unknown param — drop
        default = spec.get("default")
        choices = spec.get("choices")
        if choices:
            cleaned[k] = v if v in choices else default
            continue
        if isinstance(default, bool):
            cleaned[k] = bool(v)
            continue
        if isinstance(default, (int, float)) and isinstance(v, (int, float)) \
                and not isinstance(v, bool):
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            if isinstance(default, int):
                # Round, then re-clamp: fractional bounds (min 0.1) must not
                # be violated by the rounding itself.
                v = round(v)
                if lo is not None:
                    v = max(v, math.ceil(lo))
                if hi is not None:
                    v = min(v, math.floor(hi))
                cleaned[k] = int(v)
            else:
                cleaned[k] = float(v)
            continue
        if type(v) is type(default) or default is None:
            cleaned[k] = v
        else:
            cleaned[k] = default
    node["params"] = cleaned


def repair_graph(graph: dict, pool: GenePool | None = None,
                 cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict | None:
    """Repair a bare graph dict in place-ish (returns a new dict) or None."""
    pool = pool or build_gene_pool(cfg)

    nodes = [dict(n) for n in graph.get("nodes", [])
             if n.get("method_id") in pool.defs]
    if not nodes:
        return None
    nodes_by_id = {n["id"]: n for n in nodes}

    edges = [dict(e) for e in graph.get("edges", [])
             if _edge_legal(pool, nodes_by_id, e)]
    edges = _dagify(nodes, edges)

    # ── Terminal: exactly one render:true IMAGE producer ─────────────
    def _is_image(n: dict) -> bool:
        return "image" in (pool.defs[n["method_id"]].get("outputs") or {}).values()

    image_nodes = [n for n in nodes if _is_image(n)]
    if not image_nodes:
        return None
    depth = _topo_depth(nodes, edges)
    flagged = [n for n in image_nodes if n.get("render")]
    terminal = max(flagged or image_nodes, key=lambda n: depth[n["id"]])
    for n in nodes:
        n["render"] = n is terminal

    # ── Prune anything that can't influence the terminal ─────────────
    keep = _ancestors_of(terminal["id"], edges) | {terminal["id"]}
    nodes = [n for n in nodes if n["id"] in keep]
    edges = [e for e in edges if e["src_node"] in keep and e["dst_node"] in keep]

    for n in nodes:
        clamp_params(pool, n)

    return {**graph, "nodes": nodes, "edges": edges}


def repair_genome(genome: dict, pool: GenePool | None = None,
                  cfg: ShootoutConfig = DEFAULT_CONFIG) -> dict | None:
    """Repair a genome envelope; None means discard and resample."""
    fixed = repair_graph(genome["graph"], pool, cfg)
    if fixed is None:
        return None
    return {**genome, "graph": fixed}


def validate_graph(graph: dict, pool: GenePool | None = None,
                   cfg: ShootoutConfig = DEFAULT_CONFIG) -> list[str]:
    """Return a list of validity issues (empty == valid). Test hook — checks
    the same invariants repair guarantees."""
    pool = pool or build_gene_pool(cfg)
    issues: list[str] = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    nodes_by_id = {n["id"]: n for n in nodes}

    if len(nodes_by_id) != len(nodes):
        issues.append("duplicate node ids")
    for n in nodes:
        if n.get("method_id") not in pool.defs:
            issues.append(f"unknown method {n.get('method_id')!r}")
    for e in edges:
        if not _edge_legal(pool, nodes_by_id, e):
            issues.append(f"illegal edge {e.get('src_node')}.{e.get('src_port')}"
                          f" → {e.get('dst_node')}.{e.get('dst_port')}")

    # Cycle check (non-feedback edges)
    kept = _dagify(nodes, [e for e in edges])
    if len(kept) != len(edges):
        issues.append("cycle in non-feedback edges")

    terminals = [n for n in nodes if n.get("render")]
    if len(terminals) != 1:
        issues.append(f"{len(terminals)} render-flagged nodes (want 1)")
    elif "image" not in (pool.defs.get(terminals[0]["method_id"], {})
                         .get("outputs") or {}).values():
        issues.append("terminal is not an IMAGE producer")

    # Param ranges
    for n in nodes:
        d = pool.defs.get(n.get("method_id"))
        if d is None:
            continue
        for k, v in (n.get("params") or {}).items():
            spec = (d.get("params") or {}).get(k)
            if not isinstance(spec, dict):
                continue
            choices = spec.get("choices")
            if choices and v not in choices:
                issues.append(f"{n['id']}.{k}={v!r} not in choices")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if spec.get("min") is not None and v < spec["min"]:
                    issues.append(f"{n['id']}.{k}={v} < min {spec['min']}")
                if spec.get("max") is not None and v > spec["max"]:
                    issues.append(f"{n['id']}.{k}={v} > max {spec['max']}")
    return issues


def sample_valid_genome(pool: GenePool, cfg: ShootoutConfig,
                        rng: random.Random, origin: str = "random",
                        max_tries: int = 20) -> dict:
    """random_genome + repair, resampling until valid (plan §7 last bullet)."""
    from .generator import random_genome
    for _ in range(max_tries):
        g = repair_genome(random_genome(pool, cfg, rng, origin=origin), pool, cfg)
        if g is not None:
            return g
    raise RuntimeError("could not sample a repairable genome")
