"""Evolution engine — mutation + crossover over rated parents (plan §8).

Every offspring runs through repair before it leaves this module, so the
output of next_generation() is always render-ready. Anything unrepairable
falls back to a fresh random genome (keeps the pool full).
"""
from __future__ import annotations

import random
import uuid

from .config import ShootoutConfig, DEFAULT_CONFIG
from .generator import (
    GenePool, SamplingBias, build_gene_pool, new_genome_id, random_genome,
    sample_params, _auto_layout, _fillable_ports, _pick_producer,
)
from .repair import repair_genome


# ── Helpers ───────────────────────────────────────────────────────────


def _numeric_ranged_params(pool: GenePool, method_id: str) -> dict[str, dict]:
    out = {}
    for k, spec in (pool.defs[method_id].get("params") or {}).items():
        if not isinstance(spec, dict):
            continue
        d = spec.get("default")
        if isinstance(d, (int, float)) and not isinstance(d, bool) \
                and spec.get("min") is not None and spec.get("max") is not None:
            out[k] = spec
    return out


def _same_output_type(pool: GenePool, method_id: str) -> list[str]:
    """Method ids sharing the primary output type set (swap candidates)."""
    my_types = frozenset(pool.defs[method_id]["outputs"].values())
    cands = []
    for mid in (pool.image_producers if "image" in my_types
                else pool.producers_by_type.get(next(iter(my_types), ""), [])):
        if mid == method_id:
            continue
        if my_types <= set(pool.defs[mid]["outputs"].values()) | {"scalar", "field"}:
            cands.append(mid)
    return cands


def _remap_ids(nodes: list[dict], edges: list[dict],
               suffix: str) -> tuple[list, list, dict[str, str]]:
    mapping = {n["id"]: f"{n['id']}{suffix}" for n in nodes}
    new_nodes = [dict(n, id=mapping[n["id"]]) for n in nodes]
    new_edges = [dict(e, src_node=mapping[e["src_node"]],
                      dst_node=mapping[e["dst_node"]])
                 for e in edges
                 if e["src_node"] in mapping and e["dst_node"] in mapping]
    return new_nodes, new_edges, mapping


def _ancestor_subtree(graph: dict, root_id: str) -> tuple[list[dict], list[dict]]:
    """root node + all its ancestors + the edges between them."""
    back: dict[str, list[str]] = {}
    for e in graph["edges"]:
        back.setdefault(e["dst_node"], []).append(e["src_node"])
    keep = {root_id}
    frontier = [root_id]
    while frontier:
        cur = frontier.pop()
        for p in back.get(cur, ()):
            if p not in keep:
                keep.add(p)
                frontier.append(p)
    nodes = [dict(n) for n in graph["nodes"] if n["id"] in keep]
    edges = [dict(e) for e in graph["edges"]
             if e["src_node"] in keep and e["dst_node"] in keep]
    for n in nodes:
        n["render"] = False
    return nodes, edges


# ── Mutation operators (plan §8) ─────────────────────────────────────


def _op_param_jitter(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                     rng: random.Random) -> None:
    node = rng.choice(graph["nodes"])
    ranged = _numeric_ranged_params(pool, node["method_id"])
    params = node.setdefault("params", {})
    for k, spec in ranged.items():
        if rng.random() > 0.35:
            continue
        lo, hi = spec["min"], spec["max"]
        cur = params.get(k, spec.get("default", lo))
        if not isinstance(cur, (int, float)) or isinstance(cur, bool):
            continue
        v = cur + rng.gauss(0, cfg.param_jitter_sigma * (hi - lo))
        v = min(max(v, lo), hi)
        params[k] = int(round(v)) if isinstance(spec.get("default"), int) else round(v, 4)
    # occasional enum flip
    for k, spec in (pool.defs[node["method_id"]].get("params") or {}).items():
        if isinstance(spec, dict) and spec.get("choices") and rng.random() < 0.15:
            params[k] = rng.choice(spec["choices"])


def _op_node_swap(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                  rng: random.Random) -> None:
    node = rng.choice(graph["nodes"])
    cands = _same_output_type(pool, node["method_id"])
    if node.get("render"):
        cands = [m for m in cands if "image" in pool.defs[m]["outputs"].values()]
    if not cands:
        return
    new_mid = rng.choice(cands)
    node["method_id"] = new_mid
    has_img = any(e["dst_node"] == node["id"] for e in graph["edges"])
    node["params"] = sample_params(pool, cfg, rng, new_mid, has_img)


def _op_add_driver(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                   rng: random.Random) -> None:
    if not pool.scalar_drivers:
        return
    targets = [(n, p) for n in graph["nodes"]
               for p in pool.wireable_params(n["method_id"])
               if p not in cfg.frozen_params]
    if not targets:
        return
    tgt, param = rng.choice(targets)
    mid = rng.choice(pool.scalar_drivers)
    nid = f"d{uuid.uuid4().hex[:6]}"
    graph["nodes"].append({
        "id": nid, "method_id": mid,
        "params": sample_params(pool, cfg, rng, mid, False),
        "x": 0, "y": 0, "render": False,
    })
    graph["edges"].append({
        "src_node": nid,
        "src_port": pool.output_port_for(mid, "scalar") or "value",
        "dst_node": tgt["id"], "dst_port": param,
    })


def _op_remove_node(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                    rng: random.Random) -> None:
    victims = [n for n in graph["nodes"] if not n.get("render")]
    if not victims:
        return
    victim = rng.choice(victims)
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != victim["id"]]
    graph["edges"] = [e for e in graph["edges"]
                      if victim["id"] not in (e["src_node"], e["dst_node"])]


def _op_insert_filter(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                      rng: random.Random) -> None:
    """Grow the backbone: splice an image-processing node into an existing
    image edge (src → [new filter] → dst), or feed an unfed image port."""
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    filters = [m for m in pool.image_producers
               if any(t == "image" for _, t in _fillable_ports(pool.defs[m]))]
    if not filters:
        return
    mid = rng.choice(filters)
    d = pool.defs[mid]
    in_port = next((p for p, t in _fillable_ports(d)
                    if t == "image" and p == "image_in"),
                   next(p for p, t in _fillable_ports(d) if t == "image"))
    nid = f"f{uuid.uuid4().hex[:6]}"
    new_node = {"id": nid, "method_id": mid,
                "params": sample_params(pool, cfg, rng, mid, True),
                "x": 0, "y": 0, "render": False}

    img_edges = []
    for e in graph["edges"]:
        dst = nodes_by_id.get(e["dst_node"])
        if dst is None or e["src_node"] not in nodes_by_id:
            continue
        t = (pool.defs[dst["method_id"]].get("inputs") or {}).get(e["dst_port"])
        if t == "image":
            img_edges.append(e)

    graph["nodes"].append(new_node)
    if img_edges:
        e = rng.choice(img_edges)
        graph["edges"].append({"src_node": e["src_node"], "src_port": e["src_port"],
                               "dst_node": nid, "dst_port": in_port})
        e["src_node"] = nid
        e["src_port"] = pool.output_port_for(mid, "image")
        return
    fed = {(e["dst_node"], e["dst_port"]) for e in graph["edges"]}
    open_img = [(n, p) for n in graph["nodes"] if n["id"] != nid
                for p, t in _fillable_ports(pool.defs[n["method_id"]])
                if t == "image" and (n["id"], p) not in fed]
    if open_img:
        tgt, port = rng.choice(open_img)
        graph["edges"].append({"src_node": nid,
                               "src_port": pool.output_port_for(mid, "image"),
                               "dst_node": tgt["id"], "dst_port": port})
        return
    # No image wiring anywhere (e.g. a lone scalar-input sim) — append the
    # filter AFTER the current terminal as a post-process. Always legal, so
    # every graph can grow regardless of its input ports.
    terminal = next((n for n in graph["nodes"] if n.get("render")), None)
    if terminal is None or terminal is new_node:
        graph["nodes"].pop()
        return
    graph["edges"].append({"src_node": terminal["id"],
                           "src_port": pool.output_port_for(
                               terminal["method_id"], "image") or "image",
                           "dst_node": nid, "dst_port": in_port})
    terminal["render"] = False
    new_node["render"] = True


def _op_add_branch(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                   rng: random.Random) -> None:
    """Grow sideways: feed a random unfed structural port (field/mask/
    particles/image) with a fresh producer node."""
    fed = {(e["dst_node"], e["dst_port"]) for e in graph["edges"]}
    open_ports = [(n, p, t) for n in graph["nodes"]
                  for p, t in _fillable_ports(pool.defs[n["method_id"]])
                  if (n["id"], p) not in fed]
    if not open_ports:
        return
    tgt, port, ptype = rng.choice(open_ports)
    mid = _pick_producer(pool, cfg, rng, ptype, leaf_only=True)
    if mid is None:
        return
    nid = f"b{uuid.uuid4().hex[:6]}"
    graph["nodes"].append({"id": nid, "method_id": mid,
                           "params": sample_params(pool, cfg, rng, mid, False),
                           "x": 0, "y": 0, "render": False})
    graph["edges"].append({"src_node": nid,
                           "src_port": pool.output_port_for(mid, ptype),
                           "dst_node": tgt["id"], "dst_port": port})


def _op_rewire(graph: dict, pool: GenePool, cfg: ShootoutConfig,
               rng: random.Random) -> None:
    if not graph["edges"]:
        return
    edge = rng.choice(graph["edges"])
    src = next(n for n in graph["nodes"] if n["id"] == edge["src_node"])
    src_type = pool.defs[src["method_id"]]["outputs"].get(edge["src_port"])
    if src_type is None:
        return
    # candidate (node, port) targets accepting src_type
    targets = []
    for n in graph["nodes"]:
        if n["id"] == edge["src_node"]:
            continue
        d = pool.defs[n["method_id"]]
        for p, t in (d.get("inputs") or {}).items():
            if pool.accepts(src_type, t):
                targets.append((n["id"], p))
    if targets:
        edge["dst_node"], edge["dst_port"] = rng.choice(targets)


_MUTATION_OPS = [
    (_op_param_jitter, 3.0),
    (_op_node_swap, 2.0),
    (_op_insert_filter, 1.5),   # growth ops — favorites gain structure
    (_op_add_branch, 1.5),      # across generations (no size cap)
    (_op_add_driver, 1.0),
    (_op_remove_node, 0.7),
    (_op_rewire, 1.0),
]

# Gentle mode: structure is protected (advisor "keep this, tweak params").
_GENTLE_OPS = [(_op_param_jitter, 1.0)]


def mutate(parent: dict, pool: GenePool, cfg: ShootoutConfig,
           rng: random.Random, generation: int,
           gentle: bool = False) -> dict | None:
    """1–2 mutation ops + occasional seed jitter → repaired child or None."""
    import copy
    graph = copy.deepcopy(parent["graph"])
    op_table = _GENTLE_OPS if gentle else _MUTATION_OPS
    lo, hi = cfg.mutations_per_offspring
    ops = rng.choices([op for op, _ in op_table],
                      weights=[w for _, w in op_table],
                      k=rng.randint(lo, hi))
    for op in ops:
        op(graph, pool, cfg, rng)
    seed = parent.get("seed", 42)
    if rng.random() < 0.2:  # seed jitter (occasional — seed is genome-carried)
        seed = rng.randint(0, 2**31 - 1)

    gid = new_genome_id()
    graph["name"] = gid
    _auto_layout(graph["nodes"], graph["edges"])
    child = {
        "genome_id": gid,
        "generation": generation,
        "parents": [parent["genome_id"]],
        "origin": "mutation",
        "seed": seed,
        "graph": graph,
        "render": None, "liveness": None, "rating": None,
    }
    return repair_genome(child, pool, cfg)


def crossover(parent_a: dict, parent_b: dict, pool: GenePool,
              cfg: ShootoutConfig, rng: random.Random,
              generation: int) -> dict | None:
    """Splice a subgraph of B into A at a port-type-compatible boundary."""
    import copy
    graph = copy.deepcopy(parent_a["graph"])

    # Boundary: an existing edge in A (replace its feed), else an open
    # structural input port on some A node.
    boundary = None  # (dst_node_id, dst_port, port_type)
    if graph["edges"] and rng.random() < 0.7:
        e = rng.choice(graph["edges"])
        dst = next(n for n in graph["nodes"] if n["id"] == e["dst_node"])
        ptype = (pool.defs[dst["method_id"]].get("inputs") or {}).get(e["dst_port"])
        if ptype:
            boundary = (e["dst_node"], e["dst_port"], ptype)
    if boundary is None:
        open_ports = []
        fed = {(e["dst_node"], e["dst_port"]) for e in graph["edges"]}
        for n in graph["nodes"]:
            for p, t in (pool.defs[n["method_id"]].get("inputs") or {}).items():
                if t in ("image", "field", "mask", "particles") \
                        and (n["id"], p) not in fed:
                    open_ports.append((n["id"], p, t))
        if not open_ports:
            return None
        boundary = rng.choice(open_ports)

    dst_id, dst_port, ptype = boundary

    # Donor from B: node whose output feeds the boundary type, plus ancestors.
    donors = [n for n in parent_b["graph"]["nodes"]
              if pool.output_port_for(n["method_id"], ptype) is not None]
    if not donors:
        return None
    donor = rng.choice(donors)
    sub_nodes, sub_edges = _ancestor_subtree(parent_b["graph"], donor["id"])
    sub_nodes, sub_edges, mapping = _remap_ids(sub_nodes, sub_edges,
                                               f"x{uuid.uuid4().hex[:4]}")
    donor_id = mapping[donor["id"]]

    # Drop A's old feed into the boundary port, splice B's subtree in.
    graph["edges"] = [e for e in graph["edges"]
                      if not (e["dst_node"] == dst_id and e["dst_port"] == dst_port)]
    graph["nodes"].extend(sub_nodes)
    graph["edges"].extend(sub_edges)
    graph["edges"].append({
        "src_node": donor_id,
        "src_port": pool.output_port_for(donor["method_id"], ptype),
        "dst_node": dst_id, "dst_port": dst_port,
    })

    gid = new_genome_id()
    graph["name"] = gid
    _auto_layout(graph["nodes"], graph["edges"])
    child = {
        "genome_id": gid,
        "generation": generation,
        "parents": [parent_a["genome_id"], parent_b["genome_id"]],
        "origin": "crossover",
        "seed": rng.choice([parent_a, parent_b]).get("seed", 42),
        "graph": graph,
        "render": None, "liveness": None, "rating": None,
    }
    return repair_genome(child, pool, cfg)


# ── Generation composition ────────────────────────────────────────────


def select_parents(rated: list[dict], cfg: ShootoutConfig) -> tuple[list[dict], list[float]]:
    """Rating-weighted parent pool (4–5★ dominate; below threshold never breed)."""
    parents = [g for g in rated
               if isinstance(g.get("rating"), (int, float))
               and g["rating"] >= cfg.min_rating_to_parent]
    weights = [(g["rating"] / 5.0) ** 2 for g in parents]
    return parents, weights


def next_generation(rated: list[dict], generation: int,
                    pool: GenePool | None = None,
                    cfg: ShootoutConfig = DEFAULT_CONFIG,
                    rng: random.Random | None = None,
                    guidance: dict | None = None) -> list[dict]:
    """Compose the next candidate pool: exploit (mutation/crossover of
    rating-weighted parents) + explore (fresh randoms). Advisor guidance
    (from user notes) drops/protects specific parents and biases all fresh
    sampling. Returns cfg.render_pool unrendered, repaired genomes."""
    from .advisor import bias_from_guidance
    from .repair import sample_valid_genome
    pool = pool or build_gene_pool(cfg)
    rng = rng or random.Random()

    guidance = guidance or {}
    bias = bias_from_guidance(guidance)
    protect = set(guidance.get("protect_genomes") or [])
    drop = set(guidance.get("drop_genomes") or [])

    breedable = [g for g in rated if g["genome_id"] not in drop]
    parents, weights = select_parents(breedable, cfg)
    n_total = cfg.render_pool
    n_explore = max(1, round(cfg.explore_ratio * n_total)) if parents else n_total

    out: list[dict] = []
    while len(out) < n_total - n_explore:
        child = None
        if len(parents) >= 2 and rng.random() < cfg.crossover_ratio:
            pa, pb = rng.choices(parents, weights=weights, k=2)
            if pa is pb:
                pb = rng.choices(parents, weights=weights, k=1)[0]
            if pa["genome_id"] in protect:   # protected structure: no splicing
                child = mutate(pa, pool, cfg, rng, generation, gentle=True)
            else:
                child = crossover(pa, pb, pool, cfg, rng, generation)
        if child is None and parents:
            parent = rng.choices(parents, weights=weights, k=1)[0]
            child = mutate(parent, pool, cfg, rng, generation,
                           gentle=parent["genome_id"] in protect)
        if child is None:
            child = sample_valid_genome(pool, cfg, rng, origin="random", bias=bias)
            child["generation"] = generation
        out.append(child)

    while len(out) < n_total:
        g = sample_valid_genome(pool, cfg, rng, origin="explorer", bias=bias)
        g["generation"] = generation
        out.append(g)

    return out
