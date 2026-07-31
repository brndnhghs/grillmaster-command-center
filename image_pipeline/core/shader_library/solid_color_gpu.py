"""solid_color_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("solid_color_gpu", "Solid color fill (typed color picker)",
          "procedural", '''
void main() {
    f_color = vec4(u_color, 1.0);
}
''', uniforms={
    "color": {"glsl": "color", "default": "#4a9eff", "description": "fill color"},
})