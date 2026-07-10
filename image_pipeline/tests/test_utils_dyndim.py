"""Regression tests for the ``_DynDim`` canvas-proxy NumPy interop.

``W`` / ``H`` are ``_DynDim`` proxies (from ``image_pipeline.core.utils``) whose
value resolves dynamically from the active canvas ``ContextVar``.  Any node can
pass them straight into NumPy functions (``np.arange(H)``, ``np.mgrid[:H, :W]``,
``np.zeros((H, W))`` …), which makes NumPy call the object's ``__array__`` hook.

NumPy 2.0 changed the protocol to ``__array__(dtype, copy=False)`` (and even
stricter in 2.4, where ``np.array(int, copy=False)`` hard-raises).  A ``_DynDim``
that only accepts ``dtype`` triggered a ``DeprecationWarning`` on every such
call — and would eventually break outright.  These tests pin the contract:

* ``__array__`` must accept the 2-tuple call signature (and ignore ``copy``),
* no ``DeprecationWarning`` may surface when a proxy feeds a NumPy function,
* the resolved value/shape must be correct.
"""

import warnings

import numpy as np
import pytest

from image_pipeline.core.utils import W, H, set_canvas


@pytest.fixture
def canvas():
    token = set_canvas(64, 96)  # (W, H) = (64, 96)
    yield (64, 96)
    from image_pipeline.core.utils import reset_canvas

    reset_canvas(token)


def test_dyndim_array_no_deprecation(canvas):
    """Passing W/H into a NumPy function must not emit DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        arr_h = np.arange(H)
        arr_w = np.arange(W)
    assert arr_h.shape == (canvas[1],)
    assert arr_w.shape == (canvas[0],)


def test_dyndim_array_values_resolve(canvas):
    """The proxy resolves to the live canvas dimensions, not stale import-time 0."""
    w, h = canvas
    assert int(np.asarray(W)) == w
    assert int(np.asarray(H)) == h
    # mgrid / zeros-style consumption
    grid = np.mgrid[0:H, 0:W]
    assert grid.shape[1:] == (h, w)
    z = np.zeros((H, W))
    assert z.shape == (h, w)


def test_dyndim_array_dtype_roundtrip(canvas):
    """__array__ honours an explicit dtype argument."""
    out = np.asarray(W, dtype=np.float32)
    assert out.dtype == np.float32
    assert float(out) == float(canvas[0])


def test_dyndim_reshapes_when_canvas_changes():
    """Resolving is dynamic — a new set_canvas() changes what W/H produce."""
    from image_pipeline.core.utils import reset_canvas

    tok = set_canvas(32, 48)
    try:
        assert np.arange(H).shape == (48,)
    finally:
        reset_canvas(tok)
    tok2 = set_canvas(10, 20)
    try:
        assert np.arange(H).shape == (20,)
    finally:
        reset_canvas(tok2)
