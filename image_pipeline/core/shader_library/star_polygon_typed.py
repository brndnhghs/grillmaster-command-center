"""star_polygon_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("star_polygon_typed", "Star polygon {n/k}: connected vertex rosette (typed, node 287)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.12 * u_speed;
    int N = int(u_points);
    int K = int(clamp(u_skip, 1.0, float(N) - 1.0));
    float best = 1e9;
    for (int i = 0; i < 240; i++) {
        if (i >= N) break;
        float a0 = (float(i) / float(N)) * 6.28318530 + t;
        float a1 = (float((i + K) % N) / float(N)) * 6.28318530 + t;
        vec2 v0 = vec2(cos(a0), sin(a0)) * u_scale;
        vec2 v1 = vec2(cos(a1), sin(a1)) * u_scale;
        vec2 d = v1 - v0;
        float l2 = max(dot(d, d), 1e-6);
        float h = clamp(dot(p - v0, d) / l2, 0.0, 1.0);
        best = min(best, length(p - (v0 + d * h)));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, inferno(clamp(length(p) / max(u_scale,1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "points": {"glsl": "int", "min": 5, "max": 40, "default": 12,
               "description": "vertex count n"},
    "skip":   {"glsl": "int", "min": 2, "max": 20, "default": 5,
               "description": "step k ({n/k})"},
    "scale":  {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.9,
               "description": "polygon radius"},
    "thick":  {"glsl": "float", "min": 0.004, "max": 0.06, "default": 0.012,
               "description": "line thickness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
    "bg":     {"glsl": "color", "default": "#05060c", "description": "background"},
})