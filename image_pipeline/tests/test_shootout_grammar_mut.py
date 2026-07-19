"""Grammar-aware mutation (Route 8 / Phase 1C sub-problem #3/#4).

Verifies the `retarget_driver` mutation op (evolve._op_retarget_driver),
gated behind ``cfg.grammar_mut_ratio``:

  (a) with grammar_mut_ratio == 0.0 the live mutation path NEVER fires the
      retarget_driver op (behaviour-preserving default — no regression),
  (b) with grammar_mut_ratio > 0.0 the op fires and every produced child is a
      valid (repairable) genome,
  (c) the op is a safe no-op on a graph that has no driver (no crash / no
      mutation of the edge set),
  (d) the op actually rewires an existing driver onto a DIFFERENT (node, port)
      target that is a valid wireable scalar port.

No rendering is performed — all checks run on genome dicts in-process.
"""
import random

from image_pipeline.shootout.config import ShootoutConfig, DEFAULT_CONFIG
from image_pipeline.shootout.generator import build_gene_pool, sample_params
from image_pipeline.shootout.repair import repair_genome, sample_valid_genome
from image_pipeline.shootout.evolve import mutate, _op_retarget_driver


def _pool():
    return build_gene_pool(DEFAULT_CONFIG)


def _ensure_driver(genome, pool, cfg, rng):
    """Return a genome guaranteed to contain a wired driver node."""
    g = genome["graph"]
    has_driver = any(
        n["method_id"] in pool.scalar_drivers
        and any(e["src_node"] == n["id"] for e in g["edges"])
        for n in g["nodes"]
    )
    if has_driver:
        return genome
    target = next((n for n in g["nodes"] if n.get("render")
                   and pool.wireable_params(n["method_id"])), None)
    if target is None:
        return genome
    mid = next(iter(pool.scalar_drivers))
    nid = f"d{rng.randint(0, 10 ** 9)}"
    g["nodes"].append({
        "id": nid, "method_id": mid,
        "params": sample_params(pool, cfg, rng, mid, False),
        "x": 0, "y": 0, "render": False,
    })
    g["edges"].append({
        "src_node": nid,
        "src_port": pool.output_port_for(mid, "scalar") or "value",
        "dst_node": target["id"],
        "dst_port": pool.wireable_params(target["method_id"])[0],
    })
    return genome


def test_grammar_off_never_fires_retarget():
    pool = _pool()
    rng = random.Random(12345)
    cfg = ShootoutConfig()
    cfg.grammar_mut_ratio = 0.0
    cfg.max_depth = 3          # small parent graph keeps mutate() fast
    cfg.max_divergence_attempts = 2
    # Disable the render-in-repair variance probe so this test stays
    # headless + fast (no full-clip renders); we only exercise the mutation
    # op's gating + graph surgery, not liveness measurement.
    cfg.terminal_variance_probe = False
    parent = _ensure_driver(sample_valid_genome(pool, cfg, rng), pool, cfg, rng)
    seen_ops = set()
    for _ in range(60):
        child = mutate(parent, pool, cfg, rng, generation=1)
        if child is None:
            continue
        seen_ops.update(child["deviation"].get("ops", []))
    assert "op_retarget_driver" not in seen_ops, \
        f"retarget_driver fired with grammar_mut_ratio=0: {seen_ops}"


def test_grammar_on_fires_retarget_and_stays_valid():
    pool = _pool()
    rng = random.Random(98765)
    cfg = ShootoutConfig()
    cfg.grammar_mut_ratio = 1.0
    cfg.max_depth = 3          # small parent graph keeps mutate() fast
    cfg.max_divergence_attempts = 2
    # Disable the render-in-repair variance probe (see test_grammar_off_*).
    cfg.terminal_variance_probe = False
    parent = _ensure_driver(sample_valid_genome(pool, cfg, rng), pool, cfg, rng)
    seen_ops = set()
    n_valid = 0
    for _ in range(80):
        child = mutate(parent, pool, cfg, rng, generation=1)
        if child is None:
            continue
        n_valid += 1
        seen_ops.update(child["deviation"].get("ops", []))
    assert n_valid > 0, "no valid children produced with grammar mutation on"
    assert "op_retarget_driver" in seen_ops, \
        f"retarget_driver never fired with grammar_mut_ratio=1: {seen_ops}"


def test_op_retarget_is_noop_without_driver():
    pool = _pool()
    rng = random.Random(42)
    cfg = ShootoutConfig()
    cfg.terminal_variance_probe = False  # headless: no full-clip renders in repair
    target_mid = next(m for m, d in pool.defs.items()
                      if "image" in d.get("outputs", {}).values()
                      and pool.wireable_params(m))
    graph = {
        "nodes": [{
            "id": "t1", "method_id": target_mid,
            "params": sample_params(pool, cfg, rng, target_mid, True),
            "x": 0, "y": 0, "render": True,
        }],
        "edges": [],
        "name": "g-nodriver",
    }
    before = [(e["dst_node"], e["dst_port"]) for e in graph["edges"]]
    _op_retarget_driver(graph, pool, cfg, rng)  # must not raise
    after = [(e["dst_node"], e["dst_port"]) for e in graph["edges"]]
    assert before == after  # no driver present -> edge set unchanged


def test_op_retarget_rewires_existing_driver():
    pool = _pool()
    rng = random.Random(7)
    cfg = ShootoutConfig()
    cfg.terminal_variance_probe = False  # headless: no full-clip renders in repair
    driver_mid = next(iter(pool.scalar_drivers))
    targets = [m for m, d in pool.defs.items()
               if "image" in d.get("outputs", {}).values()
               and pool.wireable_params(m)]
    t1, t2 = targets[0], targets[1]
    p1 = pool.wireable_params(t1)[0]
    p2 = pool.wireable_params(t2)[0]
    graph = {
        "nodes": [
            {"id": "t1", "method_id": t1,
             "params": sample_params(pool, cfg, rng, t1, True),
             "x": 0, "y": 0, "render": True},
            {"id": "t2", "method_id": t2,
             "params": sample_params(pool, cfg, rng, t2, True),
             "x": 0, "y": 0, "render": False},
            {"id": "d1", "method_id": driver_mid,
             "params": sample_params(pool, cfg, rng, driver_mid, False),
             "x": 0, "y": 0, "render": False},
        ],
        "edges": [{
            "src_node": "d1",
            "src_port": pool.output_port_for(driver_mid, "scalar") or "value",
            "dst_node": "t1", "dst_port": p1,
        }],
        "name": "g-retarget",
    }
    original = ("t1", p1)
    _op_retarget_driver(graph, pool, cfg, rng)
    e = graph["edges"][0]
    new_target = (e["dst_node"], e["dst_port"])
    assert new_target != original, \
        f"retarget_driver left the driver on the same (node,port) {original}"
    nd = next(n for n in graph["nodes"] if n["id"] == new_target[0])
    assert new_target[1] in pool.wireable_params(nd["method_id"]), \
        f"retargeted to a non-wireable port {new_target}"
    # the rewired graph still repairs to a valid genome when possible
    gid = "g-retarget-child"
    graph["name"] = gid
    child = repair_genome(
        {"genome_id": gid, "graph": graph, "render": None,
         "liveness": None, "rating": None, "generation": 0,
         "origin": "mutation", "parents": [], "deviation": {}},
        pool, cfg,
    )
    if child is not None:
        e2 = next(e for e in child["graph"]["edges"] if e["src_node"] == "d1")
        nd2 = next(n for n in child["graph"]["nodes"] if n["id"] == e2["dst_node"])
        assert e2["dst_port"] in pool.wireable_params(nd2["method_id"]), \
            "repair dropped/moved the driver to a non-wireable port"
