"""Spatial params must keep reaching the pixels.

A param declaring ``spatial: True`` promises that wiring a FIELD to it varies
the output per-pixel. That promise is easy to break silently — the pre-migration
codebase had three nodes reading ``_field_*`` and responding to none of it, one
of them (#01 ASCII Art) after building a genuine per-pixel array and collapsing
it with ``np.median`` two call layers down. Nothing but a render catches that.

So: for every declared spatial param, render the node with a uniform field and
with a ramp of the SAME MEAN. A node that collapses the field cannot tell them
apart and emits identical pixels; divergence is proof the structure survived.

Regenerate the full report with:
    python tools/audit_field_response.py --scan
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
from image_pipeline.core.spatial import sparam, is_field  # noqa: E402

sys.path.insert(0, str(REPO / "tools"))
from audit_field_response import probe_param  # noqa: E402

from image_pipeline.core.utils import set_canvas  # noqa: E402


def _declared_spatial() -> list[tuple[str, str]]:
    out = []
    for mid, meta in registry.get_all().items():
        for pname, spec in (meta.params or {}).items():
            if isinstance(spec, dict) and spec.get("spatial"):
                out.append((mid, pname))
    return sorted(out)


SPATIAL_PARAMS = _declared_spatial()


def test_some_params_are_spatial():
    """Guards the guard: an empty list would make every test below vacuous."""
    assert SPATIAL_PARAMS, "no params declare spatial: True — did the migration revert?"


@pytest.mark.parametrize("mid,param", SPATIAL_PARAMS, ids=lambda v: str(v))
def test_spatial_param_reaches_pixels(mid, param):
    set_canvas(96, 72)
    r = probe_param(mid, param)
    assert r["verdict"] == "SPATIAL", (
        f"{mid}.{param} declares spatial: True but probed {r['verdict']} "
        f"(Δuniform={r['d_uniform']:.6f}). The field is not reaching the pixels — "
        f"look for a mean/median collapse, or a preset that overwrites the wired "
        f"value. {r.get('error', '')}"
    )


@pytest.mark.parametrize("mid,param", SPATIAL_PARAMS, ids=lambda v: str(v))
def test_spatial_param_exposes_field_port(mid, param):
    """spatial: True must actually produce a wireable FIELD port."""
    nd = _make_node_def(registry.get_all()[mid])
    assert nd.inputs.get(param) == "field", (
        f"{mid}.{param} declares spatial: True but exposes "
        f"{nd.inputs.get(param)!r} instead of a field port"
    )
    assert param in nd.param_ports


def test_sparam_scalar_path_is_unchanged():
    """Unwired nodes must be bit-identical to the pre-migration float() read."""
    assert sparam({"feed": 0.04}, "feed", 0.035) == 0.04
    assert sparam({}, "feed", 0.035) == 0.035
    assert sparam(None, "feed", 0.035) == 0.035
    assert sparam({"feed": "nonsense"}, "feed", 0.035) == 0.035
    assert not is_field(sparam({"feed": 0.04}, "feed", 0.035))


def test_sparam_field_path_returns_the_map():
    import numpy as np
    f = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    got = sparam({"_field_feed": f}, "feed", 0.035)
    assert is_field(got) and got.shape == (3, 4)
    # int cast must NOT quantise a wired map back toward a constant
    got_i = sparam({"_field_feed": f}, "feed", 1, cast=int)
    assert is_field(got_i) and got_i.dtype == np.float32
