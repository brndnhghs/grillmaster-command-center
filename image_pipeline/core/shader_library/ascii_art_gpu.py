"""ascii_art_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ..ascii_gpu_shader import build_ascii_glsl



_register("ascii_art_gpu", "ASCII-art with shape-vector selection (6D staggered sampling + contrast enhancement + multi-font)",
          "filter", build_ascii_glsl(), uniforms={
    "cell_size": {"glsl": "float", "min": 4.0, "max": 32.0, "default": 8.0,
                  "description": "character cell width (px)"},
    "cell_aspect": {"glsl": "float", "min": 1.0, "max": 3.0, "default": 2.0,
                    "description": "cell height/width ratio (monospace ~2:1)"},
    "font":      {"glsl": "choice", "choices": ["menlo", "courier", "monaco", "sf-mono",
                                                  "andale", "courier-new"],
                  "default": "menlo", "description": "monospace font for glyph rendering"},
    "contrast": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 1.0,
                 "description": "global contrast exponent (>1 enhances edges)"},
    "directional_strength": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 1.0,
                             "description": "directional contrast exponent (>1 sharpens external edges)"},
    "charset":   {"glsl": "choice", "choices": ["full", "classic", "minimal", "dense", "letters",
                                                  "caps", "lower", "symbols", "digits", "wide",
                                                  "sharp", "half"],
                  "default": "full", "description": "character set preset"},
    "mode":      {"glsl": "choice", "choices": ["mono", "colored", "terminal"],
                  "default": "colored", "description": "coloring mode"},
    "fg_color":  {"glsl": "color", "default": "#e8e8e8", "description": "glyph color (mono mode)"},
    "bg_color":  {"glsl": "color", "default": "#0a0a10", "description": "background color"},
    "invert":    {"glsl": "int", "min": 0, "max": 1, "default": 0,
                  "description": "invert brightness ramp"},
    "gamma":     {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                  "description": "brightness gamma before ramp"},
})