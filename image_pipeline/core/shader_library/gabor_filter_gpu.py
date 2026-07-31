"""gabor_filter_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 439 Gabor Filter (client-GPU twin) ──
_register("gabor_filter_gpu",
          "Gabor Filter (client-GPU twin of node 439)",
          "filter", _filter_typed('''
    // Single-orientation Gabor kernel response (magnitude). The filter bank +
    // hue-vs-energy output modes of the CPU node are dropped; this twin shows
    // the energy magnitude of one Gabor at the chosen orientation. Orientation
    // rotates with u_time so the live preview is animated.
    int hk = int(clamp(u_sigma * 3.0 / max(u_aspect, 0.2), 3.0, 15.0));
    float theta = u_orientation + u_time * 0.3 * u_anim_speed;
    float ct = cos(theta), st = sin(theta);
    float sigma2 = 2.0 * u_sigma * u_sigma;
    vec3 acc = vec3(0.0);
    vec3 wsum = vec3(0.0);
    for (int y = -15; y <= 15; y++) {
        if (abs(y) > hk) break;
        for (int x = -15; x <= 15; x++) {
            if (abs(x) > hk) break;
            vec2 d = vec2(float(x), float(y)) / u_resolution;
            vec3 s = sample(clamp(uv + d, 0.0, 1.0)).rgb;
            float xr = float(x) * ct + float(y) * st;
            float yr = -float(x) * st + float(y) * ct;
            float env = exp(-(xr * xr + (u_aspect * yr) * (u_aspect * yr)) / sigma2);
            float k = env * cos(6.2831853 * u_frequency * xr + u_phase);
            acc += s * k;
            wsum += vec3(k);
        }
    }
    vec3 resp = acc / max(abs(wsum), vec3(1e-3));
    float mag = clamp(length(resp) * u_contrast, 0.0, 1.0);
    f_color = vec4(vec3(mag), 1.0);
'''), uniforms={
    "orientation": {"glsl": "float", "min": 0.0, "max": 3.14159, "default": 0.0,
                    "description": "filter orientation (rad)"},
    "frequency":   {"glsl": "float", "min": 0.02, "max": 0.5, "default": 0.12,
                    "description": "Gabor spatial frequency (cycles/px)"},
    "sigma":       {"glsl": "float", "min": 2.0, "max": 24.0, "default": 8.0,
                    "description": "Gaussian envelope std (px)"},
    "aspect":      {"glsl": "float", "min": 0.2, "max": 1.0, "default": 0.5,
                    "description": "envelope elongation gamma"},
    "phase":       {"glsl": "float", "min": 0.0, "max": 6.28318, "default": 0.0,
                    "description": "sinusoid phase (rad)"},
    "contrast":    {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0,
                    "description": "response contrast boost"},
    "anim_speed":  {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                    "description": "animation speed"},
})