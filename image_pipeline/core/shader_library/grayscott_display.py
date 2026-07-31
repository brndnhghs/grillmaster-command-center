"""grayscott_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("grayscott_display",
          "Gray-Scott display: V activator → grayscale (gamma 0.5, matches _render_v)",
          "procedural", '''
void main() {
''')