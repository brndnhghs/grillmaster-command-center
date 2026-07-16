"""Port-type-aware wild genome sampling.

A genome IS the existing graph JSON (nodes/edges) plus a metadata envelope —
no new format. Sampling walks backwards from a random IMAGE terminal, filling
each input port with a producer whose output type the port accepts, so graphs
are wild *within* type correctness (plan §6). `repair.py` is the safety net.
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field

from image_pipeline.core.graph import get_all_node_defs
from image_pipeline.core.port_types import all_port_types
import image_pipeline.methods  # noqa: F401 — registers the node catalog

from .config import ShootoutConfig, DEFAULT_CONFIG

# Params that are file/asset references — never sampled, never jittered.
_UNSAMPLED_NAME_FRAGMENTS = ("path", "url", "file", "prompt")


@dataclass
class SamplingBias:
    """Steers sampling without changing the gene pool itself. Built from
    user notes by advisor.extract_guidance (or left default)."""
    prefer_methods: set = field(default_factory=set)
    avoid_methods: set = field(default_factory=set)
    prefer_categories: set = field(default_factory=set)
    avoid_categories: set = field(default_factory=set)
    complexity: float = 0.0   # -1 shrink … +1 grow (skews the size budget)

    def weight(self, pool: "GenePool", method_id: str) -> float:
        d = pool.defs[method_id]
        if method_id in self.avoid_methods or \
                d.get("category") in self.avoid_categories:
            return 0.0
        w = 1.0
        if method_id in self.prefer_methods:
            w *= 4.0
        if d.get("category") in self.prefer_categories:
            w *= 2.0
        return w


def _port_compat() -> dict[str, set[str]]:
    """dst_type -> set of src_types it accepts (lowercase), from the registry.

    An edge src.out → dst.in is legal iff dst type == src type, dst lists src
    in accepts_from, or dst is ANY (plan §3).
    """
    compat: dict[str, set[str]] = {}
    for name, spec in all_port_types().items():
        dst = name.lower()
        compat[dst] = {dst} | {a.lower() for a in spec.accepts_from}
    compat.setdefault("any", set())  # ANY accepts everything — special-cased
    return compat


@dataclass
class GenePool:
    """Catalog partition by role, derived purely from node-defs (cached)."""
    defs: dict[str, dict]
    compat: dict[str, set[str]]
    terminals: list[str] = field(default_factory=list)       # IMAGE producers, terminal-eligible
    image_producers: list[str] = field(default_factory=list)  # IMAGE producers (mid-chain too)
    producers_by_type: dict[str, list[str]] = field(default_factory=dict)
    scalar_drivers: list[str] = field(default_factory=list)   # data-only (LFO, Math, …)

    def accepts(self, src_type: str, dst_type: str) -> bool:
        src_type, dst_type = src_type.lower(), dst_type.lower()
        if dst_type == "any":
            return True
        return src_type in self.compat.get(dst_type, {dst_type})

    def output_port_for(self, method_id: str, dst_type: str) -> str | None:
        """Name of an output port on method_id that dst_type accepts.

        Prefers the primary payload ports over sidecars (luminance etc.).
        """
        outs = self.defs[method_id]["outputs"]
        # Exact-type ports named like their type first ("image", "field", …)
        for pname, ptype in outs.items():
            if pname == ptype and self.accepts(ptype, dst_type):
                return pname
        for pname, ptype in outs.items():
            if self.accepts(ptype, dst_type):
                return pname
        return None

    def wireable_params(self, method_id: str) -> list[str]:
        return list(self.defs[method_id].get("param_ports") or [])

    # Executor-owned timeline/clock params. A CHOP driver may legally
    # wire to them (they are scalar), but driving them does NOT move
    # pixels: `time`/`phase` are the injected animation clock,
    # `time_scale`/`dt` are the pipeline's speed/timestep, `global_frame`
    # is the frame counter. Empirically an LFO -> time_scale yields
    # EXACT-0 frame-to-frame variance and the clip is culled as
    # static/flat. Route 8 (2026-07-14): exclude them so
    # _ensure_animated / the p_driver draw prefer a pixel-moving
    # visual/structural param (weight_scale, frequency, amplitude, ...).
    _CLOCK_PARAMS = frozenset({"time", "time_scale", "phase",
                                  "dt", "global_frame", "total_frames"})

    # Architecture-A (no "time" param) methods run an internal sim loop and
    # cache their state across frames; the executor re-injects a driver scalar
    # each frame, but these params are consumed ONCE at sim initialisation, so
    # modulating them cannot move the clip — it stays static and is culled as
    # dead. Route 8 (2026-07-14): exclude them from driver targets for Arch-A
    # methods so the born-animated floor / driver draw prefer a per-frame-live
    # param instead. (Name-hint based: heuristic but strictly conservative —
    # dropping a rarely-live target only ever removes an inert driver edge.)
    _INIT_ONLY_HINTS = (
        "seed", "n_particles", "num_particles", "particle_count",
        "n_iterations", "iterations", "n_steps", "steps", "substeps",
        "samples", "grid_div", "grid_size", "grid_w", "grid_h",
        "cell_size", "resolution", "n_points", "num_points",
        "n_agents", "population", "count", "n_cells",
    )

    def driver_targets(self, method_id: str) -> list[str]:
        """Ports a scalar driver can legally feed: auto param ports plus
        declared scalar structural inputs (speed/rate/… on sims).
        Executor-owned clock params are excluded (see _CLOCK_PARAMS); for
        Architecture-A methods (no "time" param) init-only params are also
        excluded because they are consumed once at sim init and cannot
        animate the clip."""
        d = self.defs[method_id]
        cands = self.wireable_params(method_id) + \
            [p for p, t in _declared_ports(d) if t == "scalar"]
        is_arch_a = "time" not in (d.get("params") or {})
        out = []
        for p in cands:
            if p in self._CLOCK_PARAMS:
                continue
            if is_arch_a and p in self._INIT_ONLY_HINTS:
                continue
            out.append(p)
        return out


_POOL_CACHE: dict[tuple, GenePool] = {}


def build_gene_pool(cfg: ShootoutConfig = DEFAULT_CONFIG) -> GenePool:
    """Partition the node catalog by role (plan §6.1). Pure function of
    node-defs + config exclusions, so cache it."""
    key = (cfg.exclude_categories, cfg.exclude_methods)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]

    defs = get_all_node_defs()
    pool = GenePool(defs=defs, compat=_port_compat())

    for mid, d in defs.items():
        if d.get("deprecated"):
            continue
        if d.get("category") in cfg.exclude_categories:
            continue
        if mid in cfg.exclude_methods:
            continue
        out_types = set(d["outputs"].values())
        # Skip nodes with non-pipeline port types (3d object/geometry/…)
        known = {"image", "scalar", "field", "particles", "mask", "colormap", "any"}
        if not out_types & known:
            continue

        if "image" in out_types:
            pool.image_producers.append(mid)
            pool.terminals.append(mid)
        for t in out_types & (known - {"any"}):
            pool.producers_by_type.setdefault(t, []).append(mid)
        if out_types <= {"scalar"}:
            pool.scalar_drivers.append(mid)

    _POOL_CACHE[key] = pool
    return pool


# ── Sampling ──────────────────────────────────────────────────────────


def _declared_ports(d: dict) -> list[tuple[str, str]]:
    """Non-param declared input ports (structural inputs, not auto param ports)."""
    param_ports = set(d.get("param_ports") or [])
    return [(p, t) for p, t in (d.get("inputs") or {}).items() if p not in param_ports]


_FILLABLE_TYPES = ("image", "field", "mask", "particles", "colormap")


def _fillable_ports(d: dict) -> list[tuple[str, str]]:
    """Structural ports the backward walk can feed (scalar ports are driver
    targets, not chain continuations — ~1/3 of image producers declare only
    scalar inputs and would otherwise dead-end every chain)."""
    return [(p, t) for p, t in _declared_ports(d) if t in _FILLABLE_TYPES]


def _is_needy(d: dict) -> bool:
    """True when the node is useless without wired inputs (pure combiner —
    every declared port is an a/b merge port)."""
    ports = _declared_ports(d)
    return bool(ports) and all(p.endswith(("_a", "_b")) for p, _ in ports)


def _pick_producer(pool: GenePool, cfg: ShootoutConfig, rng: random.Random,
                   dst_type: str, leaf_only: bool,
                   bias: SamplingBias | None = None,
                   prefer_continuation: bool = False) -> str | None:
    """Weighted pick of a node whose output feeds dst_type.

    prefer_continuation boosts nodes that themselves have structural input
    ports, so chains keep growing while size budget remains — without it,
    backbones die at the first pure source and graphs stall at ~3 nodes.
    """
    cands = []
    weights = []
    universe = pool.image_producers if dst_type == "image" else \
        [m for t, ms in pool.producers_by_type.items() for m in ms
         if pool.accepts(t, dst_type)]
    seen = set()
    for mid in universe:
        if mid in seen:
            continue
        seen.add(mid)
        d = pool.defs[mid]
        if pool.output_port_for(mid, dst_type) is None:
            continue
        if leaf_only and _is_needy(d):
            continue
        w = cfg.time_varying_weight if d.get("is_time_varying") else 1.0
        if prefer_continuation and _fillable_ports(d):
            w *= cfg.continuation_weight
        if bias is not None:
            w *= bias.weight(pool, mid)
        if w > 0:
            cands.append(mid)
            weights.append(w)
    if not cands:
        if bias is not None:   # avoid-lists filtered everything — relax them
            return _pick_producer(pool, cfg, rng, dst_type, leaf_only,
                                  None, prefer_continuation)
        return None
    return rng.choices(cands, weights=weights, k=1)[0]


# Base size-budget distribution (extra nodes beyond the terminal). Heavier
# mid/tail than a flat draw — the sampler should regularly reach 4–8 node
# graphs and occasionally larger. Sizes past the list get a small constant
# tail, so raising max_depth in config directly unlocks bigger graphs.
_BUDGET_BASE = [1.0, 3.0, 4.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6]


def sample_budget(cfg: ShootoutConfig, rng: random.Random,
                  complexity: float = 0.0) -> int:
    """Draw the extra-node budget; complexity in [-1, 1] tilts the whole
    distribution toward bigger (+) or smaller (-) graphs."""
    n = max(cfg.max_depth, 1)
    weights = [
        (_BUDGET_BASE[i] if i < len(_BUDGET_BASE) else 0.4)
        * math.exp(0.6 * complexity * i)
        for i in range(n)
    ]
    return rng.choices(range(n), weights=weights, k=1)[0]


def sample_params(pool: GenePool, cfg: ShootoutConfig, rng: random.Random,
                  method_id: str, has_image_input: bool) -> dict:
    """Sample a node's params from its schema (plan §6.3): numeric biased
    toward default ± spread, enums uniform, bools coin-flip, occasional
    extremes for surprise.

    Time-varying nodes: if the schema exposes an animation-mode enum that
    includes ``"none"`` (e.g. ``anim_mode``), sampling ``"none"`` freezes the
    node's internal animation entirely — so even with no driver the clip reads
    as static and gets culled by the liveness gate. We bias those picks toward
    a real (non-``none``) mode so a time-varying node is born animated (a
    driver, when present, modulates ON TOP of that). Route 8 fix (2026-07-12):
    the corpus showed ~41% of TV nodes were sampled to a frozen ``none`` mode,
    the dominant cause of fresh ``static``/``flat`` rejections.
    """
    out: dict = {}
    is_tv = bool(pool.defs[method_id].get("is_time_varying"))
    for pname, spec in (pool.defs[method_id].get("params") or {}).items():
        if not isinstance(spec, dict):
            continue
        if pname in cfg.frozen_params:
            continue
        if any(f in pname.lower() for f in _UNSAMPLED_NAME_FRAGMENTS):
            continue
        default = spec.get("default")
        choices = spec.get("choices")
        if choices:
            opts = [c for c in choices if c != "input_image" or has_image_input]
            if opts:
                # Bias a time-varying node's animation-mode enum away from the
                # frozen "none" choice so it is born animated. Applies to every
                # enum whose choices include "none" (anim_mode, animation_mode,
                # effect, glitch_type, …) — a "none" mode freezes that aspect of
                # an otherwise animation-capable node.
                if is_tv and "none" in opts:
                    live = [c for c in opts if c != "none"]
                    if live:
                        out[pname] = rng.choice(live)
                        continue
                out[pname] = rng.choice(opts)
            continue
        if isinstance(default, bool):
            out[pname] = rng.random() < 0.5
            continue
        if isinstance(default, (int, float)):
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and hi is not None and hi > lo:
                r = rng.random()
                if r < cfg.p_extreme_param:
                    val = rng.choice((lo, hi))
                elif r < 0.7:
                    mode = min(max(default, lo), hi)
                    val = rng.triangular(lo, hi, mode)
                else:
                    val = rng.uniform(lo, hi)
            else:
                # Wireable numeric (no slider range) — multiplicative jitter
                val = default * (2 ** rng.uniform(-1, 1)) if default else rng.uniform(0, 1)
            if isinstance(default, int):
                # Round, then re-clamp so fractional bounds (min 0.1) survive
                # the rounding (repair.clamp_params does the same).
                val = round(val)
                if lo is not None:
                    val = max(val, math.ceil(lo))
                if hi is not None:
                    val = min(val, math.floor(hi))
                out[pname] = int(val)
            else:
                out[pname] = round(float(val), 4)
            continue
        if isinstance(default, str):
            out[pname] = default  # free-form string — keep as-is
        # list/tuple defaults (colors, grids) — leave to the node default
    return out


def _auto_layout(nodes: list[dict], edges: list[dict]) -> None:
    """Cheap left→right layout by upstream depth so graphs open cleanly in
    the editor (plan §6.5)."""
    depth: dict[str, int] = {n["id"]: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for e in edges:
            if e["src_node"] in depth and e["dst_node"] in depth:
                want = depth[e["src_node"]] + 1
                if depth[e["dst_node"]] < want:
                    depth[e["dst_node"]] = want
                    changed = True
        if not changed:
            break
    col_rows: dict[int, int] = {}
    for n in sorted(nodes, key=lambda n: depth[n["id"]]):
        d = depth[n["id"]]
        row = col_rows.get(d, 0)
        col_rows[d] = row + 1
        n["x"] = 80 + d * 260
        n["y"] = 80 + row * 150


def new_genome_id() -> str:
    return f"g-{uuid.uuid4().hex[:8]}"


def random_graph(pool: GenePool, cfg: ShootoutConfig, rng: random.Random,
                 bias: SamplingBias | None = None) -> dict:
    """Sample one wild-but-type-plausible graph (nodes + edges)."""
    nodes: list[dict] = []
    edges: list[dict] = []
    counter = [0]

    def add_node(method_id: str, has_image_input: bool, render: bool = False) -> str:
        counter[0] += 1
        nid = f"n{counter[0]}"
        nodes.append({
            "id": nid,
            "method_id": method_id,
            "params": sample_params(pool, cfg, rng, method_id, has_image_input),
            "x": 0, "y": 0,
            "render": render,
        })
        return nid

    budget = [sample_budget(cfg, rng, bias.complexity if bias else 0.0)]

    def fill_inputs(nid: str, method_id: str) -> None:
        d = pool.defs[method_id]
        for port, ptype in _declared_ports(d):
            is_merge = port.endswith(("_a", "_b"))
            if ptype == "image":
                p = 1.0 if is_merge else cfg.p_fill_image
            elif ptype in ("field", "mask", "particles", "colormap"):
                p = 1.0 if is_merge else cfg.p_fill_aux
            else:
                continue  # scalar/any structural ports — leave to drivers
            if budget[0] <= 0 or rng.random() > p:
                continue
            budget[0] -= 1
            src_mid = _pick_producer(pool, cfg, rng, ptype,
                                     leaf_only=budget[0] <= 0, bias=bias,
                                     prefer_continuation=budget[0] > 0)
            if src_mid is None:
                budget[0] += 1
                continue
            src_port = pool.output_port_for(src_mid, ptype)
            src_id = add_node(src_mid, has_image_input=False)
            edges.append({"src_node": src_id, "src_port": src_port,
                          "dst_node": nid, "dst_port": port})
            fill_inputs(src_id, src_mid)

    terminal_mid = _pick_producer(pool, cfg, rng, "image",
                                  leaf_only=budget[0] <= 0, bias=bias,
                                  prefer_continuation=budget[0] > 0) \
        or rng.choice(pool.terminals)
    terminal_id = add_node(terminal_mid, has_image_input=False, render=True)
    fill_inputs(terminal_id, terminal_mid)

    # Mark nodes that ended up with an image feed so 'source' style enum
    # params can legally select input_image (resample those params).
    fed = {e["dst_node"] for e in edges
           if pool.defs[{n["id"]: n for n in nodes}[e["dst_node"]]["method_id"]]
           ["inputs"].get(e["dst_port"]) == "image"}
    for n in nodes:
        if n["id"] in fed:
            n["params"] = sample_params(pool, cfg, rng, n["method_id"], True)

    # Optional scalar drivers (LFO → param etc., plan §6.2). Repeated draw —
    # a complex graph can carry several animated params, not at most one.
    while pool.scalar_drivers and rng.random() < cfg.p_driver:
        targets = [(n, p) for n in list(nodes)
                   if n["method_id"] not in pool.scalar_drivers
                   for p in pool.driver_targets(n["method_id"])
                   if p not in cfg.frozen_params
                   and not any(e["dst_node"] == n["id"] and e["dst_port"] == p
                               for e in edges)]
        if not targets:
            break
        tgt_node, tgt_param = rng.choice(targets)
        drv_mid = rng.choice(pool.scalar_drivers)
        drv_port = pool.output_port_for(drv_mid, "scalar") or "value"
        drv_id = add_node(drv_mid, has_image_input=False)
        edges.append({"src_node": drv_id, "src_port": drv_port,
                      "dst_node": tgt_node["id"], "dst_port": tgt_param})

    _auto_layout(nodes, edges)
    return {"version": 1, "name": "", "nodes": nodes, "edges": edges}


def _graph_has_animation_source(graph: dict, pool: GenePool) -> bool:
    """True if the graph can possibly animate without external input.

    Three independent signals:
      * a node with ``time`` in its schema (architecture-B — GraphExecutor
        injects the animation clock every frame),
      * a node whose animation-mode enum (anim_mode / animation_mode) was
        sampled to a non-``none`` value (internal animation),
      * any driver edge feeding a scalar param port (CHOP-style modulation).
    """
    nodes = graph.get("nodes", [])
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        mid = n.get("method_id")
        d = pool.defs.get(mid)
        if not d:
            continue
        if "time" in (d.get("params") or {}):
            return True
        p = n.get("params") or {}
        for k in ("anim_mode", "animation_mode"):
            spec = (d.get("params") or {}).get(k)
            if spec and "none" in (spec.get("choices") or []):
                if p.get(k) not in (None, "none"):
                    return True
    for e in graph.get("edges", []):
        sn = by_id.get(e.get("src_node"))
        if sn and sn.get("method_id") in pool.scalar_drivers:
            return True
    return False


def _ensure_animated(graph: dict, pool: GenePool, cfg: ShootoutConfig,
                     rng: random.Random,
                     bias: SamplingBias | None = None) -> dict:
    """Hard floor on born-animated genomes (Route 8, 2026-07-14).

    The random walk can emit graphs with no animation source at all — a
    single architecture-A terminal (e.g. an escape-time fractal with
    ``is_time_varying=False``, no ``time`` param, no ``anim_mode``) simply
    cannot move unless a control node modulates one of its params. When the
    driver draw happens not to fire, the clip is *genuinely* frozen and the
    liveness gate correctly culls it as ``static`` — wasting the full render.
    We guarantee at least one animation source by wiring an LFO onto the
    terminal node's first free driver target (falling back to any node).
    Idempotent: a no-op when a source already exists.
    """
    if not cfg.guarantee_born_animated:
        return graph
    if _graph_has_animation_source(graph, pool):
        return graph
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    if not nodes or not pool.scalar_drivers:
        return graph

    # Route 8 (2026-07-15): honour the advisor's avoid-set for injected drivers.
    drivers = pool.scalar_drivers
    if bias is not None:
        drivers = [m for m in drivers
                   if m not in bias.avoid_methods
                   and pool.defs.get(m, {}).get("category")
                   not in bias.avoid_categories]
    if not drivers:
        return graph

    def _free_targets(n: dict) -> list[str]:
        return [p for p in pool.driver_targets(n["method_id"])
                if p not in cfg.frozen_params
                and not any(e["dst_node"] == n["id"] and e["dst_port"] == p
                            for e in edges)]

    term = next((n for n in nodes if n.get("render")), nodes[-1])
    targets = _free_targets(term)
    if not targets:                       # terminal saturated — try any node
        for n in nodes:
            if n["method_id"] in drivers:
                continue
            if _free_targets(n):
                term, targets = n, _free_targets(n)
                break
    if not targets:
        return graph
    drv_mid = rng.choice(drivers)
    drv_port = pool.output_port_for(drv_mid, "scalar") or "value"
    drv_id = f"n-drv-{len(nodes)}-{rng.randint(0, 9999)}"
    nodes.append({
        "id": drv_id, "method_id": drv_mid,
        "params": sample_params(pool, cfg, rng, drv_mid, False),
        "x": 0, "y": 0, "render": False,
    })
    edges.append({"src_node": drv_id, "src_port": drv_port,
                  "dst_node": term["id"], "dst_port": targets[0]})
    return {"version": graph.get("version", 1), "name": graph.get("name", ""),
            "nodes": nodes, "edges": edges}


def apply_fallback_driver_policy(pool: GenePool, cfg: ShootoutConfig,
                                 rng: random.Random, bias: SamplingBias | None,
                                 graph: dict) -> dict:
    """Run the born-animated driver policy (``apply_driver_policy`` +
    ``_terminal_animated_floor``) over a raw ``random_graph`` fallback so its
    genomes are animated exactly like ``compose_graph`` output. Failure-soft:
    returns the *unchanged* graph if the policy ever throws on a malformed
    fallback. Route 8 (2026-07-13).
    """
    try:
        from . import motifs as _m
        b = _m.Builder(pool, cfg, rng, bias)
        b.nodes = list(graph.get("nodes", []))
        b.edges = list(graph.get("edges", []))
        b._n = len(b.nodes)
        _m.apply_driver_policy(b)
        return {"version": 1, "name": "", "nodes": b.nodes, "edges": b.edges}
    except Exception:
        return graph


def random_genome(pool: GenePool | None = None,
                  cfg: ShootoutConfig = DEFAULT_CONFIG,
                  rng: random.Random | None = None,
                  origin: str = "random",
                  bias: SamplingBias | None = None,
                  motif_weights: dict[str, float] | None = None) -> dict:
    """Emit one genome envelope (plan §5) around a freshly sampled graph.

    Structure is sampled by the motif grammar (motifs.compose_graph): it
    stacks workflow motifs up to the size budget, then runs the driver
    policy so every node is animated by control nodes. Falls back to the
    legacy backward-walk (random_graph) if motif composition fails.
    """
    pool = pool or build_gene_pool(cfg)
    rng = rng or random.Random()
    gid = new_genome_id()
    graph: dict | None = None
    try:
        from . import motifs
        graph = motifs.compose_graph(pool, cfg, rng, bias, motif_weights)
    except Exception:
        graph = None
    if graph is None:
        graph = random_graph(pool, cfg, rng, bias)
        # Route 8 (2026-07-13): the random_graph *fallback* must obey the same
        # born-animated driver policy as compose_graph. Historically the
        # fallback shipped genomes that were never run through the policy, so
        # its static-rejection rate was ~2x the motif path. Run the policy so
        # fallback genomes are born animated too.
        graph = apply_fallback_driver_policy(pool, cfg, rng, bias, graph)
    # Hard floor (Route 8, 2026-07-14): guarantee at least one animation
    # source regardless of which path produced the graph, so no genome is
    # *genuinely* frozen before the liveness gate even runs.
    graph = _ensure_animated(graph, pool, cfg, rng, bias)
    graph["name"] = gid
    return {
        "genome_id": gid,
        "generation": 0,
        "parents": [],
        "origin": origin,
        "seed": rng.randint(0, 2**31 - 1),
        "graph": graph,
        "render": None,
        "liveness": None,
        "rating": None,
    }
