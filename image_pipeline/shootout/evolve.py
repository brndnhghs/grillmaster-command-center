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


# ── Graph-distance metric ────────────────────────────────────────────
# Procedural measure of how different two graphs are. Drives the
# divergence-target loop in mutate(): the breeder keeps escalating
# mutation intensity until the offspring clears cfg.min_divergence, so
# evolutions are guaranteed to be meaningfully different from the parent
# instead of near-clones. Range ~[0, 1]: 0 = identical, 1 = nothing shared.


def graph_distance(a: dict, b: dict, pool: GenePool | None = None) -> float:
    """Normalised structural + parametric distance between two graphs.

    Components (each normalised to [0,1], averaged):
      - node multiset (method ids): Jaccard over node method_id counts
      - edge multiset (typed src→dst): Jaccard over (dst_method, dst_port) pairs
      - shared-node param delta: mean per-shared-node normalised param change
    """
    ga, gb = a.get("graph", a), b.get("graph", b)
    na = ga.get("nodes", [])
    nb = gb.get("nodes", [])
    if not na and not nb:
        return 0.0
    if not na or not nb:
        return 1.0

    # Node-type multiset Jaccard (method_id counts).
    def counter(items, key):
        c = {}
        for it in items:
            c[key(it)] = c.get(key(it), 0) + 1
        return c

    ca = counter(na, lambda n: n["method_id"])
    cb = counter(nb, lambda n: n["method_id"])
    node_d = _multiset_jaccard(ca, cb)
    size_d = abs(len(na) - len(nb)) / max(len(na), len(nb))

    # Edge-type multiset Jaccard (dst method + port), shape-agnostic.
    def edge_key(e, nodes):
        dst = next((n for n in nodes if n["id"] == e["dst_node"]), None)
        return (dst["method_id"] if dst else "?", e.get("dst_port"))
    ea = counter(ga.get("edges", []), lambda e: edge_key(e, na))
    eb = counter(gb.get("edges", []), lambda e: edge_key(e, nb))
    edge_d = _multiset_jaccard(ea, eb)

    # Per-shared-node parameter delta (normalised within schema range).
    by_id_a = {n["id"]: n for n in na}
    by_id_b = {n["id"]: n for n in nb}
    shared = set(by_id_a) & set(by_id_b)
    param_d = 0.0
    if shared:
        deltas = []
        for nid in shared:
            da, db = by_id_a[nid], by_id_b[nid]
            if da["method_id"] != db["method_id"]:
                deltas.append(1.0)  # same id, different node type = total
                continue
            spec = (pool.defs[da["method_id"]].get("params") or {}) if pool else {}
            pa, pb = da.get("params", {}), db.get("params", {})
            keys = set(pa) | set(pb)
            if not keys:
                deltas.append(0.0)
                continue
            dsum = 0.0
            for k in keys:
                va, vb = pa.get(k), pb.get(k)
                if va == vb:
                    dsum += 0.0
                elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                        and isinstance(spec.get(k), dict):
                    lo, hi = spec[k].get("min"), spec[k].get("max")
                    if lo is not None and hi is not None and hi > lo:
                        dsum += abs(va - vb) / (hi - lo)
                    else:
                        dsum += 1.0
                else:
                    dsum += 1.0  # structural/enum change
            deltas.append(min(dsum / len(keys), 1.0))
        param_d = sum(deltas) / len(deltas)

    # structural weight 0.7 (type + edges + size), parametric 0.3.
    structural = 0.45 * node_d + 0.25 * edge_d + 0.30 * size_d
    return round(min(1.0, 0.7 * structural + 0.3 * param_d), 4)


def _multiset_jaccard(ca: dict, cb: dict) -> float:
    inter = sum(min(ca[k], cb[k]) for k in ca if k in cb)
    union = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in set(ca) | set(cb))
    return 0.0 if union == 0 else 1.0 - inter / union


# ── Mutation operators (plan §8) ─────────────────────────────────────


def _op_param_jitter(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                     rng: random.Random,
                     sigma_scale: float = 1.0, jitter_p: float = 0.35) -> None:
    node = rng.choice(graph["nodes"])
    ranged = _numeric_ranged_params(pool, node["method_id"])
    params = node.setdefault("params", {})
    for k, spec in ranged.items():
        if rng.random() > jitter_p:
            continue
        lo, hi = spec["min"], spec["max"]
        cur = params.get(k, spec.get("default", lo))
        if not isinstance(cur, (int, float)) or isinstance(cur, bool):
            continue
        v = cur + rng.gauss(0, cfg.param_jitter_sigma * sigma_scale * (hi - lo))
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


def _apply_op(op, graph, pool, cfg, rng, intensity: int):
    """Apply one mutated op, passing escalating intensity to the jitter op
    (more params touched + larger steps the harder we push)."""
    if op is _op_param_jitter:
        op(graph, pool, cfg, rng,
           sigma_scale=1.0 + 0.6 * intensity,
           jitter_p=min(0.35 + 0.18 * intensity, 1.0))
    else:
        op(graph, pool, cfg, rng)


def mutate(parent: dict, pool: GenePool, cfg: ShootoutConfig,
           rng: random.Random, generation: int,
           gentle: bool = False) -> dict | None:
    """Mutation with a divergence target: keep escalating mutation
    intensity (more ops + larger jitter) until the child clears
    cfg.min_divergence from the parent, or after cfg.max_divergence_attempts.
    This guarantees bred offspring are meaningfully different — not clones.

    Every attempt force-includes at least one STRUCTURAL op (node swap /
    insert / branch / driver / rewire) so the graph topology actually changes
    and survives repair_graph's terminal-reachability prune. Pure param-jitter
    attempts can be pruned back to near-clones, which is what made old evolutions
    look identical to the parent. Records the achieved `divergence` and the
    `intensity` used in the deviation dict so the UI can show how extreme the
    evolution was.
    """
    import copy
    op_table = _GENTLE_OPS if gentle else _MUTATION_OPS
    op_names = [op for op, _ in op_table]
    op_weights = [w for _, w in op_table]
    structural_ops = [op for op in op_names if op is not _op_param_jitter]

    best_child = None
    best_div = -1.0
    for attempt in range(max(1, cfg.max_divergence_attempts)):
        intensity = attempt  # 0,1,2… → more ops, wider steps
        graph = copy.deepcopy(parent["graph"])
        lo = max(cfg.mutations_per_offspring[0], 1 + intensity // 2)
        hi = cfg.mutations_per_offspring[1] + intensity
        n_ops = rng.randint(lo, hi)
        applied = []
        chosen = [rng.choices(op_names, weights=op_weights, k=1)[0]
                  for _ in range(n_ops)]
        # Guarantee at least one structural op so topology genuinely changes.
        if structural_ops and not any(op in structural_ops for op in chosen):
            chosen[0] = rng.choice(structural_ops)
        for op in chosen:
            applied.append(op.__name__.lstrip("_"))
            _apply_op(op, graph, pool, cfg, rng, intensity)
        seed = parent.get("seed", 42)
        if not gentle and rng.random() < 0.2 + 0.1 * attempt:
            seed = rng.randint(0, 2**31 - 1)

        # Don't bother repairing fully until we know this is our best shot —
        # but cheap to just repair each; repair is local graph surgery.
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
            "deviation": {"kind": "mutation", "ops": applied,
                          "parent": parent.get("genome_id")},
            "render": None, "liveness": None, "rating": None,
        }
        child = repair_genome(child, pool, cfg)
        if child is None:
            continue
        div = graph_distance(parent, child, pool)
        if div > best_div:
            best_div = div
            best_child = child
            best_child["deviation"]["divergence"] = div
            best_child["deviation"]["intensity"] = attempt
        if div >= cfg.min_divergence:
            break

    if best_child is None:
        return None
    d = best_child["deviation"]
    div = d.get("divergence", 0.0)
    if gentle:
        d["kind"] = "protected"
        d["text"] = (
            "kept structurally intact (your note said keep this) "
            f"— only its parameters were pushed ({div:.0%} different)"
        )
    else:
        d["text"] = (
            f"mutated from a rated parent "
            f"({' + '.join(d['ops']) if d['ops'] else 'param jitter'}) "
            f"— {div:.0%} different from the original"
        )
    return best_child


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
    deviation = {"kind": "crossover",
                 "text": f"spliced a {donor['method_id']} subtree from a second "
                         f"rated parent into a first parent",
                 "donor": parent_b.get("genome_id"),
                 "subtree_nodes": len(sub_nodes)}
    child = {
        "genome_id": gid,
        "generation": generation,
        "parents": [parent_a["genome_id"], parent_b["genome_id"]],
        "origin": "crossover",
        "seed": rng.choice([parent_a, parent_b]).get("seed", 42),
        "graph": graph,
        "deviation": deviation,
        "render": None, "liveness": None, "rating": None,
    }
    return repair_genome(child, pool, cfg)


# ── Generation composition ────────────────────────────────────────────


def _render_cost_discount(g: dict, cfg: ShootoutConfig) -> float:
    """Cheapness multiplier for parent-breeding weight (1.0 = neutral).

    Penalise genomes that burned a lot of render wall-time so selection
    prefers cheap-alive forms and the gene pool stops over-investing in
    topologies that hit the render_timeout_s cap (the timeout cluster).
    Gated by ``cfg.render_cost_fitness_penalty`` (<=0 disables). Genomes
    not yet rendered (no ``render.wall_s``) are treated as neutral.
    """
    p = float(getattr(cfg, "render_cost_fitness_penalty", 0.0))
    if p <= 0.0:
        return 1.0
    wall = (g.get("render") or {}).get("wall_s")
    if not isinstance(wall, (int, float)) or wall <= 0:
        return 1.0
    ref = float(getattr(cfg, "render_cost_fitness_ref_s", 300.0)) or 300.0
    return 1.0 / (1.0 + p * (float(wall) / ref))


def select_parents(rated: list[dict], cfg: ShootoutConfig) -> tuple[list[dict], list[float]]:
    """Rating-weighted parent pool (top stars dominate; below threshold never breed).

    weight = (rating/5)**cfg.parent_selection_power, then multiplied by the
    render-cost discount (_render_cost_discount) when cfg.render_cost_fitness_penalty
    > 0 — so the winning forms dominate AND cheap-alive genomes out-breed
    render-expensive ones (attacks the timeout cluster). Verified by
    test_shootout_render_cost_fitness.py.

    When ``cfg.elo_fitness_enabled`` (Route 8 / sub-problem #1, default False),
    the raw ``rating/5`` term is replaced by ``elo_fitness(genome_id)`` — a
    Bayesian Bradley-Terry lower-confidence-bound that shrinks under-observed
    genomes toward the prior so a single noisy 5-star rating cannot dominate
    the parent pool. The liveness-breed fallback still works the same way.

    When human ratings are starved the rating-eligible pool is empty and the
    evolution would collapse to fresh randoms every generation (gen-0
    stagnation). If ``cfg.liveness_breed_fallback`` is set AND there are no
    rating-eligible parents, fall back to a liveness-fitness parent pool so
    genuinely-dynamic clips can still breed and the evolution compounds.
    """
    parents = [g for g in rated
               if isinstance(g.get("rating"), (int, float))
               and g["rating"] >= cfg.min_rating_to_parent]

    if getattr(cfg, "elo_fitness_enabled", False):
        # Bradley-Terry skill path: replace raw rating with Bayesian LCB.
        from . import taste_elo
        taste_elo.invalidate_cache()  # fresh model each generation
        weights = [(taste_elo.elo_fitness(g.get("genome_id", "")) ** cfg.parent_selection_power)
                   * _render_cost_discount(g, cfg) for g in parents]
    else:
        weights = [((g["rating"] / 5.0) ** cfg.parent_selection_power) * _render_cost_discount(g, cfg) for g in parents]

    if cfg.liveness_breed_fallback and (
            not parents or len(parents) < cfg.liveness_breed_min_rated):
        # Blended liveness-breeding (Route 8 follow-up): when the rated-parent
        # pool is thin, supplement it with liveness-fitness parents so the
        # abundant liveness signal from the other alive genomes still drives
        # evolution toward dynamic clips. Rated parents already in the pool are
        # excluded so each breeder appears exactly once.
        fb, fbw = _liveness_parent_pool(
            [g for g in rated if g not in parents], cfg)
        if fb:
            if not parents:
                parents, weights = fb, fbw
            else:
                parents = parents + fb
                weights = weights + [w * cfg.liveness_breed_blend for w in fbw]
    return parents, weights


def _liveness_fitness(g: dict, cfg: ShootoutConfig) -> float:
    """Fitness proxy from the liveness stats: 0 for dead/static, up to 1 for a
    clearly-dynamic clip. Blends temporal variance and changed-pixel fraction
    (both already exclude flicker via the alive gate's frame_corr check)."""
    liv = g.get("liveness") or {}
    if not liv.get("alive"):
        return 0.0
    tvar = float(liv.get("temporal_var") or 0.0)
    mpf = float(liv.get("motion_pixel_frac") or 0.0)
    f_t = min(1.0, tvar / max(cfg.temporal_var_min * 8.0, 1e-6))
    f_m = min(1.0, mpf / 0.3)
    return 0.5 * f_t + 0.5 * f_m


def _liveness_parent_pool(rated: list[dict], cfg: ShootoutConfig) -> tuple[list[dict], list[float]]:
    """Build a parent pool from alive clips whose liveness fitness clears a
    floor, weighted by fitness**power. Only triggered when no rating-eligible
    parents exist (see select_parents)."""
    pool: list[dict] = []
    weights: list[float] = []
    for g in rated:
        fit = _liveness_fitness(g, cfg)
        if fit >= 0.15:
            pool.append(g)
            weights.append((fit ** cfg.parent_selection_power) * _render_cost_discount(g, cfg))
    return pool, weights


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

    # Coverage-aware explorer booster (Route 8 sub-problem #2 — diversity): bias
    # the fresh-randoms toward under-represented motifs using inverse-frequency
    # weights over the survivor pool. When the pool is flat this returns None and
    # the explorer samples the normal prior (fully behavior-preserving); only
    # when a few motifs dominate does it nudge explorers into uncovered niches.
    from . import motifs as _motifs
    explorer_motif_weights = _motifs.coverage_biased_weights(
        breedable if parents else [], cfg.motif_coverage_boost)

    out: list[dict] = []
    while len(out) < n_total - n_explore:
        child = None
        # Cross-breed: blend TWO distinct rated parents (winning graphs bred
        # together) with probability cfg.cross_breed_probability. Retry with
        # fresh pairs so the realized cross-breed rate tracks the setting
        # instead of silently degrading to a single-parent mutation when a
        # given splice happens to be incompatible.
        if len(parents) >= 2 and rng.random() < cfg.cross_breed_probability:
            for _ in range(3):
                pa, pb = rng.choices(parents, weights=weights, k=2)
                if pa is pb:
                    pb = rng.choices(parents, weights=weights, k=1)[0]
                if pa["genome_id"] in protect:   # protected structure: vary, don't splice
                    child = mutate(pa, pool, cfg, rng, generation, gentle=True)
                    break
                child = crossover(pa, pb, pool, cfg, rng, generation)
                if child is not None:
                    break
            # child stays None only if no compatible cross existed across tries
        if child is None and parents:
            # single-parent variation (also the fallback when cross-breeding
            # is impossible with the current parent set)
            parent = rng.choices(parents, weights=weights, k=1)[0]
            child = mutate(parent, pool, cfg, rng, generation,
                           gentle=parent["genome_id"] in protect)
        if child is None:
            child = sample_valid_genome(pool, cfg, rng, origin="random",
                                        bias=bias,
                                        motif_weights=explorer_motif_weights)
            child["generation"] = generation
            child["deviation"] = {"kind": "random",
                                  "text": "fresh random graph (no parent — pure exploration)",
                                  "ops": []}
        out.append(child)

    while len(out) < n_total:
        g = sample_valid_genome(pool, cfg, rng, origin="explorer", bias=bias,
                                motif_weights=explorer_motif_weights)
        g["generation"] = generation
        g["deviation"] = {"kind": "explorer",
                          "text": "fresh random graph (explorer — keeps variety high)",
                          "ops": []}
        out.append(g)

    return out
