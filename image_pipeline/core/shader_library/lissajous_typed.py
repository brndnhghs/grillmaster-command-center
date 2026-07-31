"""lissajous_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("lissajous_typed", "Lissajous: traced harmonic figure (typed, node 279)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float best = 1e9;
    for (int i = 0; i < 240; i++) {
        float s = float(i) / 240.0 * 6.28318530;
        vec2 q = vec2(sin(u_fx * s + u_phase + t), sin(u_fy * s)) * u_scale;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "fx":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
              "description": "x frequency"},
    "fy":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
              "description": "y frequency"},
    "phase": {"glsl": "float", "min": 0.0, "max": 6.28, "default": 1.57,
              "description": "phase offset"},
    "scale": {"glsl": "float", "min": 0.3, "max": 0.95, "default": 0.8,
              "description": "figure size"},
    "thick": {"glsl": "float", "min": 0.01, "max": 0.12, "default": 0.04,
              "description": "trace thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "drift speed"},
    "bg":    {"glsl": "color", "default": "#04060c", "description": "background"},
    "fg":    {"glsl": "color", "default": "#ffe66b", "description": "trace color"},
})