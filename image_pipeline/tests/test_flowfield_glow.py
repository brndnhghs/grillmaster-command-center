"""Regression test for TD-16: flow-field neon glow color overflow.

Palette colors are ``np.uint8``; the old ``min(255, c + 100)`` wrapped on
uint8 and emitted a ``RuntimeWarning: overflow encountered in scalar add``
(and produced the wrong (wrapped) glow color). The fix casts to float and
clips to [0, 255].
"""
import warnings
from pathlib import Path

from image_pipeline.methods.simulations.flowfield import method_flowfield


def test_neon_glow_no_overflow_warning(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method_flowfield(
            Path(tmp_path),
            7,
            params={"trail_mode": "neon", "color_mode": "velocity"},
        )
    overflow = [w for w in caught
                if "overflow" in str(w.message).lower()]
    assert not overflow, f"overflow warning emitted: {overflow}"
