"""Regression guard for the spatial-param port contract in graph.py.

The 237-case ``test_spatial_params.py`` drives ``GraphExecutor`` end-to-end and
proves a wired FIELD reaches the pixels. It does NOT assert the *port type* that
``_make_node_def`` assigns — the half of the graph.py change that decides whether
a FIELD wire is even offered in the UI. This file locks that contract so a future
refactor can't silently re-close the gate that locked out ~3128 of 3170 numeric
params (2026-07-22 audit).

Contract:
  * A param declaring ``"spatial": True`` MUST get an ``inputs[param] == "field"``
    port, REGARDLESS of min/max slider hints.
  * A param with ``min``/``max`` and no ``spatial`` flag MUST NOT appear in
    ``inputs`` at all (legacy internal-control gate stays intact).
  * ``sparam`` returns the per-pixel array when ``_field_<name>`` is present,
    else the scalar — the core/spatial.py reader the nodes depend on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_pipeline.methods  # noqa: F401,E402 — registers every method
from image_pipeline.core import registry  # noqa: E402
from image_pipeline.core.graph import _make_node_def  # noqa: E402
from image_pipeline.core.spatial import sparam, is_field  # noqa: E402


def _find_spatial() -> tuple[str, str] | None:
    for mid, meta in registry.get_all().items():
        for pname, spec in (meta.params or {}).items():
            if isinstance(spec, dict) and spec.get("spatial"):
                return mid, pname
    return None


def _find_bounded_non_spatial() -> tuple[str, str] | None:
    for mid, meta in registry.get_all().items():
        for pname, spec in (meta.params or {}).items():
            if isinstance(spec, dict) and not spec.get("spatial") \
                    and ("min" in spec or "max" in spec):
                return mid, pname
    return None


def test_spatial_param_gets_field_port():
    """A ``spatial: True`` param MUST be offered a FIELD input port."""
    found = _find_spatial()
    assert found, "no spatial param found — did the migration revert?"
    mid, pname = found
    meta = registry.get_all()[mid]
    ndef = _make_node_def(meta)
    assert ndef.inputs.get(pname) == "field", (
        f"{mid}.{pname} declares spatial:True but node-def input port is "
        f"{ndef.inputs.get(pname)!r}, not 'field'"
    )


def test_bounded_non_spatial_param_not_wireable():
    """A min/max param without spatial:True stays an internal control (no port)."""
    found = _find_bounded_non_spatial()
    assert found, "no bounded non-spatial param found"
    mid, pname = found
    meta = registry.get_all()[mid]
    ndef = _make_node_def(meta)
    assert pname not in ndef.inputs, (
        f"{mid}.{pname} has min/max and no spatial flag but leaked a port "
        f"{ndef.inputs.get(pname)!r} — the legacy gate reopened"
    )


def test_sparam_returns_array_when_field_wired():
    """sparam must hand back the per-pixel array, not collapse it to a scalar."""
    arr = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    params = {"_field_feed": arr, "feed": 0.035}
    out = sparam(params, "feed", 0.035)
    assert is_field(out), "sparam returned a scalar despite a wired _field_ key"
    assert np.array_equal(np.asarray(out, dtype=np.float32), arr)


def test_sparam_returns_scalar_when_unwired():
    """sparam must behave exactly like float(params.get(...)) when nothing wired."""
    out = sparam({"feed": 0.035}, "feed", 0.035)
    assert not is_field(out)
    assert out == 0.035
    # default path
    assert sparam(None, "feed", 0.035) == 0.035
