"""`/api/node-defs` must not mutate the @cache'd node-def contract.

``core.graph.get_all_node_defs()`` is decorated with ``@cache``, so it returns
the SAME dict object on every call. ``server.get_node_defs()`` enriches param
specs with UI ``choices`` before serving them; it previously did that by
assigning ``nd['params'] = _enrich_params(...)`` directly into that shared dict.

The effect: the first ``/api/node-defs`` request permanently rewrote 227 param
specs in core's cache. Every later consumer — the executor's port derivation,
any test — saw presentation-enriched data instead of the declared contract, and
which one you got depended on whether the endpoint had been hit yet. Test
outcomes became order-dependent on HTTP traffic.

These tests lock the separation: enrichment appears in the RESPONSE, never in
the MODEL.
"""
from __future__ import annotations

import copy

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.graph import get_all_node_defs
from image_pipeline.server import get_node_defs


def test_node_defs_endpoint_does_not_mutate_cached_defs():
    """Calling the endpoint must leave the cached defs byte-identical."""
    cached = get_all_node_defs()
    before = copy.deepcopy(cached)

    get_node_defs()
    get_node_defs()  # twice: a single-shot mutation would still be caught above

    after = get_all_node_defs()
    assert after is cached, "get_all_node_defs() should still be @cache'd"

    drifted = [
        (mid, key)
        for mid, nd in after.items()
        for key in (nd.get("params") or {})
        if nd["params"][key] != before[mid]["params"][key]
    ]
    assert not drifted, (
        f"/api/node-defs mutated {len(drifted)} cached param spec(s) — "
        f"e.g. {drifted[:3]}. Enrichment must build a copy, not write into "
        f"the shared @cache'd dict."
    )


def test_node_defs_endpoint_still_enriches_the_response():
    """The fix must not silently disable choice enrichment for the UI."""
    cached_params = {
        mid: copy.deepcopy(nd.get("params") or {})
        for mid, nd in get_all_node_defs().items()
    }

    served = get_node_defs()

    enriched = [
        (nd.get("method_id"), key)
        for nd in served.values()
        for key, spec in (nd.get("params") or {}).items()
        if isinstance(spec, dict) and "choices" in spec
        and "choices" not in (cached_params[nd["method_id"]].get(key) or {})
    ]
    assert enriched, (
        "no param specs were enriched with 'choices' — the UI select factory "
        "depends on this, so an empty result means enrichment regressed"
    )


def test_node_defs_response_is_stable_across_calls():
    """Two consecutive calls must produce equal payloads (no accumulation)."""
    assert get_node_defs() == get_node_defs()
