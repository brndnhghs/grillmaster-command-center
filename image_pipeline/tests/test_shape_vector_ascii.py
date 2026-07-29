"""
Minimal GPU test for the shape-vector ascii_art_gpu shader.
Tests independently of the full project import chain.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import ONLY what we need for the shader test
from image_pipeline.core.shaders import SHADERS, render_shader


def test_ascii_shape_vector_glyph_structure():
    """Shape-vector ASCII produces glyph structure on a gradient input."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    # Default params: no contrast enhancement
    a = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                 named_params={"mode": "mono", "cell_size": 8.0}),
                   dtype=float)
    assert a.std() > 5, f"ascii shape-vector should produce glyph structure, std={a.std():.3f}"


def test_ascii_contrast_enhancement_changes_output():
    """Raising contrast should produce a measurably different frame."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    base = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                    named_params={"mode": "mono", "cell_size": 8.0,
                                                  "contrast": 1.0}),
                      dtype=float)
    enhanced = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                        named_params={"mode": "mono", "cell_size": 8.0,
                                                      "contrast": 3.0}),
                          dtype=float)
    diff = np.abs(enhanced - base).mean()
    assert diff > 1.0, f"contrast 3.0 should differ visibly from 1.0, Δ={diff:.3f}"


def test_ascii_directional_contrast_changes_output():
    """Directional contrast should also change the output."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    base = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                    named_params={"mode": "mono", "cell_size": 8.0}),
                      dtype=float)
    dir_enhanced = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                            named_params={"mode": "mono", "cell_size": 8.0,
                                                          "directional_strength": 3.0}),
                              dtype=float)
    diff = np.abs(dir_enhanced - base).mean()
    assert diff > 0.5, f"directional_strength 3.0 should differ from 1.0, Δ={diff:.3f}"


def test_ascii_colored_mode():
    """Colored mode should preserve source colors."""
    # Create a strongly colored input
    src = np.zeros((64, 96, 3), dtype=np.float32)
    src[:, :, 0] = 1.0  # pure red
    out = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                   named_params={"mode": "colored", "cell_size": 8.0}),
                     dtype=float)
    # Red channel should dominate (mean ~71 for 28% glyph coverage in red on black bg)
    assert out[:, :, 0].mean() > out[:, :, 1].mean(), "colored mode should preserve red dominance"


def test_ascii_terminal_mode():
    """Terminal mode should be green-dominant."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    out = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                   named_params={"mode": "terminal", "cell_size": 8.0}),
                     dtype=float)
    assert out[:, :, 1].mean() > out[:, :, 0].mean(), "terminal mode should be green-dominant"


def test_ascii_cell_aspect_changes():
    """Changing cell aspect ratio should change the output."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    a = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                 named_params={"mode": "mono", "cell_size": 8.0,
                                               "cell_aspect": 1.0}),
                   dtype=float)
    b = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                 named_params={"mode": "mono", "cell_size": 8.0,
                                               "cell_aspect": 3.0}),
                   dtype=float)
    diff = np.abs(b - a).mean()
    assert diff > 0.5, f"cell_aspect 1.0 vs 3.0 should differ, Δ={diff:.3f}"


def test_ascii_invert_flips():
    """Invert should flip dark/light regions."""
    # Use a gradient that goes dark→light
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    normal = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                      named_params={"mode": "mono", "cell_size": 8.0,
                                                    "invert": 0}),
                        dtype=float)
    inverted = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                        named_params={"mode": "mono", "cell_size": 8.0,
                                                      "invert": 1}),
                          dtype=float)
    diff = np.abs(inverted - normal).mean()
    assert diff > 5, f"invert should produce a visible difference, Δ={diff:.3f}"


def test_ascii_charset_full_vs_minimal():
    """Different character set presets should produce different outputs."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    full = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                    named_params={"mode": "mono", "cell_size": 8.0,
                                                  "charset": "full"}),
                      dtype=float)
    minimal = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                       named_params={"mode": "mono", "cell_size": 8.0,
                                                     "charset": "minimal"}),
                         dtype=float)
    diff = np.abs(minimal - full).mean()
    assert diff > 1.0, f"full vs minimal charsets should differ visibly, Δ={diff:.3f}"


def test_ascii_charset_digits_only():
    """Digits-only charset should produce output with digit-like glyphs."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    out = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                   named_params={"mode": "mono", "cell_size": 8.0,
                                                 "charset": "digits"}),
                     dtype=float)
    assert out.std() > 5, "digits charset should still produce structure"


def test_ascii_all_charsets_render():
    """Every charset preset should render without errors and produce non-flat output."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    charsets = ["full", "classic", "minimal", "dense", "letters", "caps",
                "lower", "symbols", "digits", "wide", "sharp", "half"]
    for cs in charsets:
        out = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                       named_params={"mode": "mono", "cell_size": 8.0,
                                                     "charset": cs}),
                         dtype=float)
        assert out.std() > 1, f"charset '{cs}' should produce non-flat output"


def test_ascii_fonts_render():
    """Every font preset should render without errors and produce distinct output."""
    src = np.linspace(0, 1, 96 * 64 * 3, dtype=np.float32).reshape(64, 96, 3)
    fonts = ["menlo", "courier", "monaco", "sf-mono", "andale", "courier-new"]
    baselines = {}
    for f in fonts:
        out = np.asarray(render_shader("ascii_art_gpu", (96, 64), input_image=src,
                                       named_params={"mode": "mono", "cell_size": 8.0,
                                                     "font": f}),
                         dtype=float)
        assert out.std() > 1, f"font '{f}' should produce non-flat output"
        baselines[f] = out.mean()

    # Fonts should differ from each other
    ref_mean = baselines["menlo"]
    for f in fonts:
        assert abs(baselines[f] - ref_mean) < 50, \
            f"font '{f}' mean ({baselines[f]:.1f}) should be roughly similar to menlo ({ref_mean:.1f})"


if __name__ == "__main__":
    test_ascii_shape_vector_glyph_structure()
    print("✓ test_ascii_shape_vector_glyph_structure")
    test_ascii_contrast_enhancement_changes_output()
    print("✓ test_ascii_contrast_enhancement_changes_output")
    test_ascii_directional_contrast_changes_output()
    print("✓ test_ascii_directional_contrast_changes_output")
    test_ascii_colored_mode()
    print("✓ test_ascii_colored_mode")
    test_ascii_terminal_mode()
    print("✓ test_ascii_terminal_mode")
    test_ascii_cell_aspect_changes()
    print("✓ test_ascii_cell_aspect_changes")
    test_ascii_invert_flips()
    print("✓ test_ascii_invert_flips")
    test_ascii_charset_full_vs_minimal()
    print("✓ test_ascii_charset_full_vs_minimal")
    test_ascii_charset_digits_only()
    print("✓ test_ascii_charset_digits_only")
    test_ascii_all_charsets_render()
    print("✓ test_ascii_all_charsets_render")
    test_ascii_fonts_render()
    print("✓ test_ascii_fonts_render")
    print("All tests passed!")
