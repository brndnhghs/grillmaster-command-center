"""Regression test for anim_mode dropdown enrichment (Leverage workstream).

Before this change, ~40 methods that branch on an aliased anim_mode variable
(e.g. ``mode = params.get("anim_mode")``) or that kept their modes in a legacy
``options`` key exposed NO choices to the UI — the animation-mode selector was a
free-text box instead of a dropdown, so users couldn't discover or reliably
pick modes. The server's _enrich_params now derives the mode enum from each
method's own source (AST scan, alias-aware) and normalises ``options``->``choices``.
"""
import pytest
from fastapi.testclient import TestClient

import image_pipeline.methods  # noqa: F401 — register methods
from image_pipeline import server as _server
from image_pipeline.core.utils import set_canvas


@pytest.fixture(scope="module")
def client():
    set_canvas(128, 96)
    with TestClient(_server.app) as c:
        yield c


def _anim_mode_choices(defs, method_id):
    nd = defs.get(method_id)
    if not nd:
        return None
    spec = (nd.get("params") or {}).get("anim_mode")
    if not spec:
        return None
    return spec.get("choices")


def test_node_defs_enriches_aliased_anim_mode(client):
    """A method that aliases anim_mode to a local var must expose a dropdown."""
    defs = client.get("/api/node-defs").json()
    # DLA (node 80? no) — use a node known to alias. Find ANY node whose
    # derived choices come from the AST path (no pre-existing choices/options).
    enriched = 0
    for mid, nd in defs.items():
        spec = (nd.get("params") or {}).get("anim_mode")
        if not spec:
            continue
        ch = spec.get("choices")
        # Skip nodes that already declared choices/options in their decorator —
        # those are covered by data-wins; we assert the DERIVED ones now exist.
        raw = _server.registry.get_all()[mid].params.get("anim_mode", {})
        if "choices" in raw or "options" in raw:
            continue
        # This node relied purely on the AST derivation.
        assert isinstance(ch, list) and len(ch) >= 2, (
            f"node {mid} {nd['name']} did not derive anim_mode choices: {ch}"
        )
        assert "none" in ch, f"node {mid} choices must include default 'none': {ch}"
        enriched += 1
    assert enriched >= 30, f"expected >=30 AST-derived anim_mode nodes, got {enriched}"


def test_options_key_normalised_to_choices(client):
    """Node 152 keeps modes in legacy 'options' — must surface as 'choices'."""
    defs = client.get("/api/node-defs").json()
    ch = _anim_mode_choices(defs, "152")
    assert ch is not None and "binary_orbit" in ch and "driven" in ch, (
        f"node 152 options not normalised: {ch}"
    )


def test_derive_helper_handles_alias():
    """_derive_anim_mode_choices must follow `mode = params.get('anim_mode')`."""
    meta = _server.registry.get_all()["57"]  # Slit Scan — aliases to `mode`
    ch = _server._derive_anim_mode_choices(meta.fn)
    assert ch and "none" in ch and "drift" in ch, f"Slit Scan derivation failed: {ch}"


def test_explicit_choices_win_over_derivation():
    """A node that already declares choices must keep them (data wins)."""
    meta = _server.registry.get_all()["402"]  # Kaleidoscopic IFS — explicit
    raw = meta.params["anim_mode"]
    assert "choices" in raw
    ch = _server._enrich_params(meta.params, meta.fn)["anim_mode"]["choices"]
    assert ch == raw["choices"], "explicit choices were overwritten by derivation"
