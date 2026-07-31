"""newton_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _TYPED_FRACTAL_HELPERS



_register("newton_typed", "Newton fractal (z^3-1) basins with typed zoom/palette (node 241)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 z = uv * (2.2 / max(u_zoom, 0.001));
    float n = 0.0;
    const int CAP = 80;
    for (int i = 0; i < CAP; i++) {
        vec2 z2 = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        vec2 z3 = vec2(z2.x*z.x - z2.y*z.y, 2.0*z2.x*z.y);
        vec2 f = z3 - vec2(1.0, 0.0);
        vec2 dz = 3.0 * z2;
        float denom = dz.x*dz.x + dz.y*dz.y + 1e-8;
        vec2 stp = vec2(f.x*dz.x + f.y*dz.y, f.y*dz.x - f.x*dz.y) / denom;
        z -= stp; n += 1.0;
        if (dot(stp, stp) < 1e-6) break;
    }
    float ang = atan(z.y, z.x);
    float root = floor((ang + 3.14159) / (2.0 * 3.14159 / 3.0));
    float t = mod(root / 3.0 + u_color_offset + 0.15 * n / 80.0, 1.0);
    vec3 col = (u_palette == 1) ? inferno_l(t * (0.6 + 0.4 * u_color_speed))
              : (u_palette == 2) ? mix(u_color_a, u_color_b, clamp(t, 0.0, 1.0))
              : fractal_palette(t * (0.6 + 0.4 * u_color_speed));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "zoom":         {"glsl": "float", "min": 0.01, "max": 8.0, "default": 1.0,
                     "description": "zoom (1 = full view)"},
    "color_speed":  {"glsl": "float", "min": 0.0, "max": 2.0, "default": 1.0,
                     "description": "color cycling speed"},
    "color_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                     "description": "color offset"},
    "palette":      {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                     "default": "sine", "description": "color palette"},
    "color_a":      {"glsl": "color", "default": "#05010a",
                     "description": "color A (grayscale)"},
    "color_b":      {"glsl": "color", "default": "#ffd166",
                     "description": "color B (grayscale)"},
})