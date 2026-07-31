"""maurer_rose_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("maurer_rose_typed", "Maurer rose: polygonal line sculpture (typed, node 285)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.08 * u_speed;
    float best = 1e9;
    int N = int(u_steps);
    float d = 3.14159265 / 180.0 * u_deg;
    for (int i = 0; i < 720; i++) {
        if (i >= N) break;
        float k = float(i);
        float ang = k * d + t;
        float rr = u_scale * sin(u_petals * ang);
        vec2 q = vec2(cos(ang), sin(ang)) * rr;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, inferno(clamp(length(p) / max(u_scale,1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "petals": {"glsl": "float", "min": 2.0, "max": 20.0, "default": 6.0,
               "description": "rose petal count"},
    "deg":    {"glsl": "float", "min": 1.0, "max": 180.0, "default": 29.0,
               "description": "connector angle (deg)"},
    "steps":  {"glsl": "int", "min": 60, "max": 720, "default": 360,
               "description": "vertex count"},
    "scale":  {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.9,
               "description": "flower radius"},
    "thick":  {"glsl": "float", "min": 0.004, "max": 0.06, "default": 0.015,
               "description": "line thickness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
    "bg":     {"glsl": "color", "default": "#05060c", "description": "background"},
})