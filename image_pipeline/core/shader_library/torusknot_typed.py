"""torusknot_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("torusknot_typed", "Torus knot ribbon: parametric (p,q) knot (typed, node 288)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.2 * u_speed;
    float best = 1e9;
    int N = int(u_steps);
    for (int i = 0; i < 600; i++) {
        if (i >= N) break;
        float s = float(i) / float(N) * 6.28318530;
        float r = cos(u_q * s) + u_rad;
        vec2 q = vec2(sin(u_p * s + t) * r, cos(u_p * s + t) * r) * u_scale;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    float hue = clamp(atan(p.y, p.x) / 6.28318530 + 0.5, 0.0, 1.0);
    vec3 col = mix(u_bg, inferno(hue), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "p":     {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
              "description": "knot winding p"},
    "q":     {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
              "description": "knot winding q"},
    "rad":   {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8,
              "description": "tube offset"},
    "steps": {"glsl": "int", "min": 120, "max": 600, "default": 400,
              "description": "curve resolution"},
    "scale": {"glsl": "float", "min": 0.2, "max": 0.8, "default": 0.45,
              "description": "knot size"},
    "thick": {"glsl": "float", "min": 0.01, "max": 0.12, "default": 0.035,
              "description": "ribbon thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "rotation speed"},
    "bg":    {"glsl": "color", "default": "#04060c", "description": "background"},
})