"""Regression test for TD-19: Langton palette symbol import.

Commit ba45df7 deduplicated the extended-palette table into
``particles.py`` and imported it from ``langtons_ant.py``. This locks that
the symbol is actually bound at module load (a dangling reference there
crashed the Langton node at runtime).
"""
import image_pipeline.methods.simulations.langtons_ant as la
import image_pipeline.methods.simulations.particles as pa


def test_langton_palette_symbol_imported():
    assert hasattr(la, "_LANGTON_EXTRA_PALETTES")
    assert hasattr(pa, "_LANGTON_EXTRA_PALETTES")
    # The table must be non-empty and shared (same object) by both modules.
    assert pa._LANGTON_EXTRA_PALETTES
    assert la._LANGTON_EXTRA_PALETTES is pa._LANGTON_EXTRA_PALETTES
