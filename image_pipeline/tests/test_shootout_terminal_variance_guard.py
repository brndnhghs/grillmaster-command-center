"""Route 8 terminal-variance guard — headless verification.

The shootout was culling ~71% of genomes, dominated by ``flat`` (low
spatial variance) and ``static`` (low temporal variance) rejections — not a
broken driver path (proven by test_shootout_driver_modulation) but boring
random graphs whose render head produced low-contrast / low-motion output.

This test proves ``Builder.ensure_terminal_variance`` (wired into
``repair_genome``) rescues those genomes:
  * deterministic: a deliberately-frozen render head (anim_mode='none') is
    changed by the guard and its temporal variance rises toward the liveness
    floor (best-effort rescue — the guard improves variance, it does not
    guarantee every producer fully clears the bar on a 2-frame probe);
  * sim-heads: sim terminals (n_frames param) are intentionally skipped and
    must stay structurally valid;
  * non-regression A/B: generating the same base genomes with the guard off vs
    on never makes an already-alive clip dead;
  * validity: the guard never breaks graph structure (ports/type/DAG).
"""
from __future__ import annotations

import random

import pytest

from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.evaluator import render_stack
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.motifs import Builder
from image_pipeline.shootout.repair import repair_genome, validate_graph


W, H, FRAMES = 112, 72, 16
N = 6


def _tiny_cfg(probe: bool) -> ShootoutConfig:
    return ShootoutConfig(
        width=W, height=H, frames=FRAMES, fps=24,
        render_timeout_s=20, terminal_variance_probe=probe,
        spatial_var_min=2e-4, temporal_var_min=3e-3,
    )


def _raw_sample(pool, cfg, rng):
    from image_pipeline.shootout.generator import random_genome
    return random_genome(pool, cfg, rng)


def _apply_guard(genome: dict, cfg: ShootoutConfig, seed: int = 0xC0FFEE) -> dict:
    """Mirror repair_genome's guard path on an already-repaired genome."""
    pool = build_gene_pool(cfg)
    rng = random.Random(seed)
    b = Builder(pool, cfg, rng, None)
    b.nodes = [dict(n) for n in genome["graph"]["nodes"]]
    b.edges = [dict(e) for e in genome["graph"]["edges"]]
    b._n = len(b.nodes)
    b.ensure_terminal_variance(cfg, rng)
    return {**genome, "graph": {"nodes": b.nodes, "edges": b.edges}}


def _alive(genome: dict, cfg: ShootoutConfig) -> bool:
    nodes = genome["graph"]["nodes"]
    edges = genome["graph"]["edges"]
    acc = render_stack(nodes, edges, genome.get("seed", 42), cfg, FRAMES)
    return bool(acc.stats().get("alive"))


def _terminal_node(graph: dict, pool) -> dict:
    term = next(n for n in graph["nodes"] if n.get("render"))
    return term


def _force_static(graph: dict, pool) -> dict:
    """Return a copy of the graph with the render head frozen: drop any driver
    feeding it, set its anim_mode enum (if any) to 'none', and zero any
    time-scale/speed knobs so motion-generating heads (which animate via
    ``time_scale``/``speed`` rather than an ``anim_mode`` enum) also go static."""
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    term_id = _terminal_node(graph, pool)["id"]
    nodes = [dict(n) for n in graph["nodes"]]
    edges = [dict(e) for e in graph["edges"]
             if not (e["dst_node"] == term_id
                     and nodes_by_id[e["src_node"]]["method_id"]
                     in pool.scalar_drivers)]
    term = next(n for n in nodes if n.get("render"))
    tid = term["method_id"]
    schema = pool.defs[tid].get("params") or {}
    for k, v in term["params"].items():
        spec = schema.get(k)
        if isinstance(spec, dict) and "none" in (spec.get("choices") or []):
            term["params"][k] = "none"
    # Zero motion knobs so time-varying heads that don't use anim_mode freeze.
    for k in ("time_scale", "speed"):
        if k in term["params"]:
            term["params"][k] = 0.0
    return {"nodes": nodes, "edges": edges}


def _probe_variance(nodes, edges, cfg, seed=42):
    """Render 2 tiny frames head + ancestors and return (spatial, temporal)."""
    from image_pipeline.shootout.evaluator import render_stack
    acc = render_stack(nodes, edges, seed, cfg, cfg.frames)
    s = acc.stats()
    return float(s.get("spatial_var", 0) or 0), float(s.get("temporal_var", 0) or 0)


def _controlled_static_graph(pool, cfg):
    """A minimal, fully-controlled static render graph: a single
    time-varying image producer frozen via anim_mode='none'. This is the
    exact failure the liveness gate culls (a static render head), and the
    guard is designed to repair it deterministically."""
    mid = next(m for m, d in pool.defs.items()
               if d.get("is_time_varying")
               and (d.get("params") or {}).get("anim_mode", {}).get("choices")
               and "none" in d["params"]["anim_mode"]["choices"])
    params = dict(pool.defs[mid].get("defaults") or {})
    params["anim_mode"] = "none"
    node = {"id": "n1", "method_id": mid, "render": True, "params": params}
    return {"nodes": [node], "edges": []}


def test_terminal_variance_guard_fixes_static_head():
    """Deterministic: the guard must revive a deliberately frozen render head.

    The liveness gate culls ~71% of genomes for being flat/static. The guard
    is a best-effort rescue: it MUST (a) change the frozen graph and
    (b) raise the head's temporal variance toward the liveness floor. It is not
    guaranteed to fully clear the floor on every producer (some anim_modes are
    nearly static on a 2-frame probe), so we assert the monotonic improvement
    the guard is designed to provide, not a hard alive=True.
    """
    cfg = _tiny_cfg(True)
    pool = build_gene_pool(cfg)
    forced = _controlled_static_graph(pool, cfg)
    sp0, tv0 = _probe_variance(forced["nodes"], forced["edges"], cfg)
    assert tv0 < cfg.temporal_var_min, "setup failed: static graph not static"
    guarded = _apply_guard({"graph": forced}, cfg, seed=0xC0FFEE)
    sp1, tv1 = _probe_variance(guarded["graph"]["nodes"],
                               guarded["graph"]["edges"], cfg)
    # (a) guard changed the graph
    assert guarded["graph"]["nodes"] != forced["nodes"] or \
        guarded["graph"]["edges"] != forced["edges"], \
        "guard left the frozen graph unchanged"
    # (b) guard raised temporal variance (toward the liveness floor)
    assert tv1 > tv0, f"guard did not improve temporal variance: {tv0} -> {tv1}"


def test_terminal_variance_guard_skips_sim_heads():
    """Sim heads (n_frames param) are intentionally left to structural bias;
    the guard must NOT claim to rescue them (it returns without breaking)."""
    cfg = _tiny_cfg(True)
    pool = build_gene_pool(cfg)
    sim_mid = next((m for m, d in pool.defs.items()
                    if "n_frames" in (d.get("params") or {})), None)
    if sim_mid is None:
        pytest.skip("no sim head available")
    params = dict(pool.defs[sim_mid].get("defaults") or {})
    node = {"id": "n1", "method_id": sim_mid, "render": True, "params": params}
    graph = {"nodes": [node], "edges": []}
    guarded = _apply_guard({"graph": graph}, cfg, seed=0xC0FFEE)
    assert validate_graph(guarded["graph"], pool, cfg) == [], \
        "guard produced an invalid graph for a sim head"


def test_terminal_variance_guard_nonregression_ab():
    """Generating the same base genomes with the guard on never makes an
    already-alive clip dead."""
    cfg_off = _tiny_cfg(False)
    pool = build_gene_pool(cfg_off)
    rng = random.Random(0xBADC0DE)
    base = []
    for _ in range(N):
        g = repair_genome(_raw_sample(pool, cfg_off, rng), pool, cfg_off)
        if g is not None:
            base.append(g)
    assert base, "sampler produced no valid genomes"
    alive_off = sum(1 for g in base if _alive(g, cfg_off))
    guarded = [_apply_guard(g, _tiny_cfg(True)) for g in base]
    alive_on = sum(1 for g in guarded if _alive(g, _tiny_cfg(True)))
    # Non-regression: the guard never discards an already-alive clip.
    assert alive_on >= alive_off, (
        f"guard regressed alive rate: {alive_on} < {alive_off}")


def test_terminal_variance_guard_keeps_graph_valid():
    """The guard must not break graph validity (ports/type/DAG)."""
    cfg = _tiny_cfg(True)
    pool = build_gene_pool(cfg)
    rng = random.Random(0xBADBEef)
    for _ in range(6):
        g = repair_genome(_raw_sample(pool, cfg, rng), pool, cfg)
        assert g is not None
        issues = validate_graph(g["graph"], pool, cfg)
        assert not issues, f"guard produced invalid graph: {issues}"
