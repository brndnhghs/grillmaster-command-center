"""Fast (no-render) regression: motif-path TV terminals must be born animated.

Route 8 (2026-07-14): the motif composer ``compose_graph`` -> ``apply_driver_policy``
-> ``_terminal_animated_floor`` must guarantee every time-varying (TV) terminal is
either driver-driven or has a live (non-``none``) animation mode. A TV terminal
with no driver and a frozen ``none`` mode renders a frozen clip and is culled as
``static``/``flat``.

The motif path is currently only exercised by the SLOW ``test_tv_terminals_born_animated``
(which renders 400 genomes via ``sample_valid_genome`` and is therefore excluded
from the default CI suite via the ``slow`` marker). The FALLBACK path is guarded
by ``test_fallback_path_born_animated``. This test closes the gap: it calls
``compose_graph`` directly (no render) across a large sample so the motif-path
born-animated invariant is enforced on EVERY run and a regression in the motif
composer's driver policy is caught immediately, not only when the slow suite runs.
"""
from __future__ import annotations

import random

import pytest

import image_pipeline.methods  # noqa: F401 — registers the node catalog
from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool
from image_pipeline.shootout.motifs import compose_graph

CFG = ShootoutConfig()
POOL = build_gene_pool(CFG)

# control-node method ids (drivers) — edges whose src is one of these
_DRIVERS = {
    "__lfo__", "__counter__", "__noise1d__", "__ramp__",
    "__strobe__", "__envelope__", "__image_to_mask__",
}

# Fraction of fresh motif-path genomes we tolerate as trivially static before
# flaking the test. The fix drives this to ~0; we allow a tiny margin for
# method ids whose TV flag is mis-set or whose only mode enum has no live
# alternative.
_MAX_TRIVIALLY_STATIC_FRAC = 0.02


def _is_tv(mid: str) -> bool:
    return bool((POOL.defs.get(mid) or {}).get("is_time_varying"))


def _has_none_mode(mid: str, params: dict) -> bool:
    schema = (POOL.defs.get(mid) or {}).get("params") or {}
    for p, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        choices = spec.get("choices")
        if choices and "none" in choices and params.get(p, "none") == "none":
            return True
    return False


def _terminal_static_risk(graph: dict) -> bool:
    """True if the terminal is TV, not driver-driven, and has a frozen mode."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    term = max(graph["nodes"], key=lambda n: n.get("render", False))
    tmid = term["method_id"]
    if not _is_tv(tmid):
        return False
    driven = any(
        nodes.get(e["src_node"], {}).get("method_id") in _DRIVERS
        and e["dst_node"] == term["id"]
        for e in edges
    )
    if driven:
        return False
    return _has_none_mode(tmid, term.get("params", {}))


def test_motif_path_born_animated():
    """compose_graph must never ship a TV terminal that is undriven AND frozen."""
    rng = random.Random(20260714)
    n = 400
    static_risk = 0
    tv_terminals = 0
    for _ in range(n):
        g = compose_graph(POOL, CFG, rng)
        term = max(g["nodes"], key=lambda nd: nd.get("render", False))
        if _is_tv(term["method_id"]):
            tv_terminals += 1
        if _terminal_static_risk(g):
            static_risk += 1
    frac = static_risk / max(tv_terminals, 1)
    assert tv_terminals > 0, "composer produced no TV terminals (unexpected)"
    assert frac <= _MAX_TRIVIALLY_STATIC_FRAC, (
        f"{static_risk}/{tv_terminals} motif-path TV terminals are trivially "
        f"static ({frac:.1%} > {_MAX_TRIVIALLY_STATIC_FRAC:.0%}): a TV terminal "
        f"with no driver and a frozen 'none' mode renders static and gets "
        f"culled. compose_graph -> apply_driver_policy -> _terminal_animated_floor "
        f"must guarantee a live animation mode on undriven TV terminals."
    )
