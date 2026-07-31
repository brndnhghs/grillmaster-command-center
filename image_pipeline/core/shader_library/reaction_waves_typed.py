"""reaction_waves_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("reaction_waves_typed", "Autonomous reaction-diffusion wave pattern (typed, node 267)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float t = u_time * 0.05;
    float v = 0.0;
    // Layered concentric reaction fronts from jittered seed centers.
    for (int i = 0; i < 8; i++) {
        float fi = float(i);
        vec2 seed = (vec2(hash21(vec2(fi, 3.1)), hash21(vec2(fi, 7.7))) - 0.5) * res;
        float d = distance(p, seed);
        float k = (u_wavelength * (1.0 + 0.25 * sin(fi * 1.7)));
        float ph = (d / k) - t * (u_speed * (1.0 + 0.15 * cos(fi * 2.3)));
        v += (0.5 + 0.5 * sin(ph * 6.2831853));
    }
    v = (v / 8.0 - 0.5) * u_contrast + 0.5;
    v = clamp(v, 0.0, 1.0);
    f_color = vec4(inferno(v), 1.0);
}
''', uniforms={
    "speed":       {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "wave propagation speed"},
    "wavelength":  {"glsl": "float", "min": 4.0, "max": 120.0, "default": 38.0,
                    "description": "front spacing (px)"},
    "contrast":    {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.0,
                    "description": "band sharpness"},
})