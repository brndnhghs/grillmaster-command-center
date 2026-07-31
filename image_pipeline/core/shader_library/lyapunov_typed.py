"""lyapunov_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _TYPED_FRACTAL_HELPERS



_register("lyapunov_typed", "Lyapunov exponent map with typed r-range/palette (node 243)",
          "procedural", _TYPED_FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    float rx = mix(u_r_min, u_r_max, uv.x);
    float ry = mix(u_r_min, u_r_max, uv.y);
    float lambda = 0.0; float x = 0.5;
    const float WARM = 30.0; const float MEAS = 120.0;
    for (float i = 0.0; i < (WARM + MEAS); i += 1.0) {
        int k = int(mod(i, 8.0));
        float rk = (k == 0 || k == 2 || k == 4 || k == 6) ? rx : ry;
        float deriv = rk * (1.0 - 2.0 * x);
        x = rk * x * (1.0 - x);
        if (i >= WARM) lambda += log(abs(deriv) + 1e-8);
    }
    lambda /= MEAS;
    float t = clamp(0.5 + 0.5 * lambda / 2.0, 0.0, 1.0);
    t = fract(t + u_color_shift);
    vec3 col = (u_palette == 2) ? mix(u_color_a, u_color_b, t)
              : (u_palette == 1) ? inferno_l(t)
              : fractal_palette(t);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "r_min":       {"glsl": "float", "min": 0.0, "max": 4.0, "default": 2.5,
                    "description": "r min (AB row)"},
    "r_max":       {"glsl": "float", "min": 0.0, "max": 4.0, "default": 4.0,
                    "description": "r max (AB row)"},
    "palette":     {"glsl": "choice", "choices": ["sine", "inferno", "grayscale"],
                    "default": "sine", "description": "color palette"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "palette shift"},
    "color_a":     {"glsl": "color", "default": "#05010a",
                    "description": "color A (grayscale)"},
    "color_b":     {"glsl": "color", "default": "#ffd166",
                    "description": "color B (grayscale)"},
})