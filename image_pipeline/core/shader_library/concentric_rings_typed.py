"""concentric_rings_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("concentric_rings_typed", "Smooth concentric rings / ripples (typed, node 270)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    p += (vec2(u_center_x, u_center_y) - 0.5) * res;
    p = rot(u_skew) * p;
    float r = length(p);
    float t = u_time * 0.05 * u_speed;
    float rings = 0.5 + 0.5 * sin(r / max(u_spacing, 1.0) * 6.2831853 - t * 6.2831853);
    rings = pow(rings, u_sharpness);
    f_color = vec4(inferno(clamp(rings, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "spacing":     {"glsl": "float", "min": 4.0, "max": 120.0, "default": 28.0,
                    "description": "ring spacing (px)"},
    "speed":       {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "ripple expansion speed"},
    "sharpness":   {"glsl": "float", "min": 0.3, "max": 6.0, "default": 1.0,
                    "description": "band sharpness"},
    "skew":        {"glsl": "float", "min": -1.5707963, "max": 1.5707963, "default": 0.0,
                    "description": "ellipse skew (rad)"},
    "center_x":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "center x"},
    "center_y":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "center y"},
})