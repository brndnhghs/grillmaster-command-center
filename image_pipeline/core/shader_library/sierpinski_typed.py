"""sierpinski_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _TYPED_FRACTAL_HELPERS



_register("sierpinski_typed", "Sierpinski carpet with typed depth/palette (node 242)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 p = v_uv;
    float depth = clamp(floor(u_depth), 1.0, 7.0);
    float hole = 0.0;
    for (float i = 0.0; i < 7.0; i += 1.0) {
        if (i >= depth) break;
        vec2 cell = floor(p * 3.0);
        if (cell.x == 1.0 && cell.y == 1.0) { hole = 1.0; break; }
        p = fract(p * 3.0);
    }
    float t = fract(0.15 * depth + u_color_shift + 0.3 * v_uv.x + 0.2 * v_uv.y);
    vec3 col = (hole > 0.5) ? u_color_a
             : _fractalColor(t, u_palette, u_color_a, u_color_b, u_color_shift);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "depth":       {"glsl": "int", "min": 1, "max": 7, "default": 4,
                    "description": "subdivision depth"},
    "palette":     {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                    "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":     {"glsl": "color", "default": "#0a0a12",
                    "description": "hole color"},
    "color_b":     {"glsl": "color", "default": "#ffd166",
                    "description": "color B (grayscale)"},
})