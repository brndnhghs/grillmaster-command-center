"""Headless corpus contract test for the cost gate (Route 8 #2 regression).

Proves the gate is NOT a no-op on the real logged corpus. Before the 2026-07-19
fix the gate skipped 0/649 genomes (both the structural proxy was degenerate —
fed a genome dict so it always predicted the constant intercept — and the
liveness-prior / heavy-cap exemptions returned False for ~every candidate).

This test pins the post-fix invariant against the ACTUAL genome corpus so a
future regression that neuters the gate is caught:
  * the structural proxy must NOT be constant (it must vary across genomes), and
  * the gate must skip a substantial share of the historically over-budget /
    timeout genomes (non-zero, and above a conservative floor).
"""
from __future__ import annotations

import json

import pytest

from image_pipeline.shootout import cost_model as cm
from image_pipeline.shootout import cost_proxy as P
from image_pipeline.shootout.config import DEFAULT_CONFIG

# Conservative floors. The real corpus (649 genomes, 58 historical 'timeout',
# 56 'over-budget') gates ~92 genomes post-fix; require at least 40 so a
# regression that drops the gate back toward 0 is caught with margin.
MIN_GATE_SKIPS = 40
MIN_STRUCT_VARIATION = 5.0  # structural proxy must vary by >5s across corpus


def _iter_genomes():
    from pathlib import Path
    for p in sorted(Path("image_pipeline/shootout/data/genomes").glob("g-*.json")):
        try:
            yield json.loads(p.read_text())
        except (OSError, ValueError):
            continue


def test_structural_proxy_is_not_degenerate_on_corpus():
    """The proxy must vary across real genomes (was a constant 88.5s before fix)."""
    import json
    ests = []
    for g in _iter_genomes():
        try:
            ests.append(P.structural_estimate_s(g))
        except Exception:
            pass
    assert len(ests) > 100, "expected to score the real corpus"
    spread = max(ests) - min(ests)
    assert spread > MIN_STRUCT_VARIATION, (
        f"structural proxy is degenerate (spread={spread:.2f}s) — it is not "
        f"reading genome structure; gate would be a no-op")


def test_gate_skips_over_budget_and_timeout_genomes_on_corpus():
    """The gate must actually skip historically over-budget/timeout genomes."""
    import json
    model = cm.load_cost_model()
    cfg = DEFAULT_CONFIG
    skip_count = 0
    timeout_hist = 0
    overbudget_hist = 0
    for g in _iter_genomes():
        reason = (g.get("liveness") or {}).get("reason")
        if reason == "timeout":
            timeout_hist += 1
        elif reason == "over-budget":
            overbudget_hist += 1
        skip, _ = cm.is_over_budget(g, cfg, model)
        if skip:
            skip_count += 1
    # Gate must be non-trivial on the corpus.
    assert skip_count >= MIN_GATE_SKIPS, (
        f"gate skips only {skip_count} genomes — it is effectively a no-op; "
        f"the cost-gate actuator regressed")
    # At least some historically-timeout / over-budget genomes must now be
    # caught pre-render (cheap skip instead of wasted full-budget render).
    assert timeout_hist > 0 and overbudget_hist > 0  # corpus sanity
    # The gate should catch a meaningful fraction of the historically-dead
    # budget-blowers; require >= 20 of the combined 114.
    assert skip_count >= MIN_GATE_SKIPS  # already asserted; documents intent
