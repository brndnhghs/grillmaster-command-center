"""mandelbrot_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _TYPED_FRACTAL_HELPERS



_register("mandelbrot_typed", "Mandelbrot set with typed zoom/center/iter/palette (node 238)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(u_center_x, u_center_y) + uv * (3.0 / max(u_zoom, 0.001));
    vec2 z = vec2(0.0);
    float n = 0.0; float last2 = 0.0;
    const int CAP = 500;
    for (int i = 0; i < CAP; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z); n += 1.0;
        if (last2 > 16.0 || n >= float(u_max_iter)) break;
    }
    float t = (n >= float(u_max_iter) - 0.5) ? 1.0 : smooth_iter(n, last2, float(u_max_iter));
    f_color = vec4(_fractalColor(t, u_palette, u_color_a, u_color_b, u_color_shift), 1.0);
}
''', uniforms={
    "zoom":       {"glsl": "float", "min": 0.01, "max": 8.0, "default": 1.0,
                   "description": "zoom (1 = full view)"},
    "center_x":   {"glsl": "float", "min": -2.0, "max": 0.5, "default": -0.5,
                   "description": "center X"},
    "center_y":   {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.0,
                   "description": "center Y"},
    "max_iter":   {"glsl": "int", "min": 20, "max": 500, "default": 200,
                   "description": "max iterations"},
    "palette":    {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                   "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":    {"glsl": "color", "default": "#05010a",
                   "description": "color A (grayscale / holes)"},
    "color_b":    {"glsl": "color", "default": "#ffd166",
                   "description": "color B (grayscale)"},
})