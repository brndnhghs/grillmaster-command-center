"""Regression test: time-varying terminals must be born animated.

Route 8 (2026-07-12): the shootout corpus was dominated by ``static``/
``flat`` rejections. Two real sampler defects caused *fresh* (post-driver
policy) genomes to render frozen:

  1. ``sample_params`` sampled a TV node's animation-mode enum (``anim_mode``,
     ``effect``, ``glitch_type`` …) uniformly — incl. ``"none"`` — ~41% of the
     time, which freezes the node's internal animation even with a driver.
  2. A TV terminal could end up with NO control-node driver AND a frozen
     ``none`` mode (p_drive_primary < 1), a guaranteed static cull.

Fix: bias TV nodes away from ``none`` modes, and add a safety net in
``apply_driver_policy`` (``_terminal_animated_floor``) that flips any
undriven TV node's frozen mode to a live choice. This test locks both in:
across many fresh gen-0 genomes, every TV terminal is either driver-driven or
has a live (non-``none``) animation mode — i.e. the trivially-static risk is
~0. It is purely structural (no rendering), so it runs in well under the
cron time budget.

Mirrors test_shootout_driver_modulation.py (which guards the SCALAR->param
injection path) but at the sampling layer.
"""
from __future__ import annotations

import random

import image_pipeline.methods  # noqa: F401 — registers the node catalog
from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.generator import build_gene_pool, DEFAULT_CONFIG
from image_pipeline.shootout.repair import sample_valid_genome

CFG = ShootoutConfig()
POOL = build_gene_pool(CFG)

# control-node method ids (drivers) — edges whose src is one of these
_DRIVERS = {
    "__lfo__", "__counter__", "__noise1d__", "__ramp__",
    "__strobe__", "__envelope__", "__image_to_mask__",
}

# fraction of fresh gen-0 genomes we tolerate as trivially static before
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


def _terminal_static_risk(genome: dict) -> bool:
    """True if the terminal is TV, not driver-driven, and has a frozen mode."""
    nodes = {n["id"]: n for n in genome["graph"]["nodes"]}
    edges = genome["graph"]["edges"]
    term = max(genome["graph"]["nodes"], key=lambda n: n.get("render", False))
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


def test_tv_terminals_born_animated():
    rng = random.Random(20260712)
    n = 400
    static_risk = 0
    tv_terminals = 0
    for _ in range(n):
        g = sample_valid_genome(POOL, CFG, rng)
        term = max(g["graph"]["nodes"], key=lambda n: n.get("render", False))
        if _is_tv(term["method_id"]):
            tv_terminals += 1
        if _terminal_static_risk(g):
            static_risk += 1
    frac = static_risk / max(tv_terminals, 1)
    assert tv_terminals > 0, "sampler produced no TV terminals (unexpected)"
    assert frac <= _MAX_TRIVIALLY_STATIC_FRAC, (
        f"{static_risk}/{tv_terminals} TV terminals are trivially static "
        f"({frac:.1%} > {_MAX_TRIVIALLY_STATIC_FRAC:.0%}): a TV terminal with "
        f"no driver and a frozen 'none' mode renders static and gets culled. "
        f"The sampler must bias TV nodes away from 'none' modes and the "
        f"driver policy must guarantee a live mode on undriven TV terminals."
    )
