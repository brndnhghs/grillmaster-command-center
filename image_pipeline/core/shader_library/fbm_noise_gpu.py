"""fbm_noise_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("fbm_noise_gpu", "Fractal Brownian motion noise with typed octave controls",
          "procedural", '''
float fbm_typed(vec2 p) {
    float v = 0.0, amp = 0.5, freq = 1.0, norm = 0.0;
    for (int i = 0; i < 10; i++) {
        if (i >= u_octaves) break;
        v += amp * noise(p * freq);
        norm += amp;
        freq *= u_lacunarity;
        amp *= u_gain;
    }
    return norm > 0.0 ? v / norm : 0.0;
}

void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    p += u_time * u_drift * vec2(0.31, 0.17);
    if (u_warp > 0.001) {
        vec2 q = vec2(fbm_typed(p + vec2(5.2, 1.3)), fbm_typed(p + vec2(8.3, 2.8)));
        p += u_warp * 4.0 * (q - 0.5);
    }
    float t = clamp(fbm_typed(p), 0.0, 1.0);
    t = pow(t, max(u_contrast, 0.05));
    f_color = vec4(mix(u_color_a, u_color_b, t), 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 0.5, "max": 32.0, "default": 6.0,
                   "description": "noise scale"},
    "octaves":    {"glsl": "int", "min": 1, "max": 10, "default": 5,
                   "description": "fbm octaves"},
    "gain":       {"glsl": "float", "min": 0.1, "max": 0.9, "default": 0.5,
                   "description": "per-octave gain"},
    "lacunarity": {"glsl": "float", "min": 1.2, "max": 4.0, "default": 2.0,
                   "description": "per-octave frequency multiplier"},
    "warp":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "domain warp amount"},
    "drift":      {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.2,
                   "description": "animation drift speed"},
    "contrast":   {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                   "description": "output contrast (gamma)"},
    "color_a":    {"glsl": "color", "default": "#06080f", "description": "low color"},
    "color_b":    {"glsl": "color", "default": "#d8e8ff", "description": "high color"},
})