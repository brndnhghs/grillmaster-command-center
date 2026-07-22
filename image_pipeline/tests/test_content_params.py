"""TEXT ports must actually deliver their string.

The counterpart to ``test_spatial_params.py``. A param declaring
``content: True`` gets a TEXT port in the UI, which promises an upstream node
can drive it — a Text Source feeding a QR payload, typography copy, GLSL source
or a font path. This checks the promise end to end by wiring a real graph.

What this gate does and does not police
--------------------------------------
It fails on ``NOT_WIRED``: the param responds when set directly but not when
wired, which is a routing defect and squarely this contract's problem.

It does NOT fail on ``INERT`` or ``ERROR``. Those mean the param does nothing
even without a wire, or the node cannot run here at all — a missing optional
dependency (``qrcode``, ``pyfiglet``), an absent binary (Blender), or a payload
only valid on a specific machine (a real image path). Failing on those would
make the suite hostage to what happens to be installed, and would point at the
wrong layer besides. They are reported, not enforced.

Regenerate the full report with:
    python tools/audit_content_response.py --scan
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: F401,E402 — registers every method
from image_pipeline.core import registry  # noqa: E402
from image_pipeline.core.graph import _make_node_def  # noqa: E402
from image_pipeline.core.port_types import get_port_type  # noqa: E402
from image_pipeline.core.utils import set_canvas  # noqa: E402

sys.path.insert(0, str(REPO / "tools"))
from audit_content_response import probe_param, TEXT_SOURCE  # noqa: E402


def _declared_content() -> list[tuple[str, str]]:
    out = []
    for mid, meta in registry.get_all().items():
        if mid == TEXT_SOURCE:
            continue
        for pname, spec in (meta.params or {}).items():
            if isinstance(spec, dict) and spec.get("content"):
                out.append((mid, pname))
    return sorted(out)


CONTENT_PARAMS = _declared_content()


def test_text_port_type_is_registered():
    spec = get_port_type("TEXT")
    assert spec is not None, "TEXT port type missing from the registry"
    assert spec.color


def test_some_params_declare_content():
    """Guards the guard: an empty list makes every test below vacuous."""
    assert CONTENT_PARAMS, "no params declare content: True"


def test_text_source_emits_text():
    nd = _make_node_def(registry.get_all()[TEXT_SOURCE])
    assert nd.outputs.get("text") == "text", "Text Source must emit a TEXT port"


@pytest.mark.parametrize("mid,param", CONTENT_PARAMS, ids=lambda v: str(v))
def test_content_param_exposes_text_port(mid, param):
    nd = _make_node_def(registry.get_all()[mid])
    assert nd.inputs.get(param) == "text", (
        f"{mid}.{param} declares content: True but exposes "
        f"{nd.inputs.get(param)!r} instead of a text port"
    )
    assert param in nd.param_ports


@pytest.mark.parametrize("mid,param", CONTENT_PARAMS, ids=lambda v: str(v))
def test_content_param_is_not_misrouted(mid, param):
    """The wire must not be the broken link.

    NOT_WIRED is the one verdict this gate owns: the param demonstrably works
    when set directly, so a wire failing to deliver is a routing bug.
    """
    set_canvas(96, 72)
    r = probe_param(mid, param)
    assert r["verdict"] != "NOT_WIRED", (
        f"{mid}.{param} responds to a direct value (Δ={r.get('direct_delta', 0):.6f}) "
        f"but not to a wired one (Δ={r['delta']:.6f}) — the TEXT wire is not reaching it"
    )
