"""harmonograph_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("harmonograph_typed", "Harmonograph: decaying Lissajous trace (typed, node 284)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.3 * u_speed;
    float best = 1e9;
    int N = int(u_steps);
    for (int i = 0; i < 400; i++) {
        if (i >= N) break;
        float s = float(i) / float(N) * 6.28318530 * u_turns;
        float env = exp(-u_decay * float(i) / float(N));
        vec2 q = vec2(
            sin(u_fx * s + u_px + t) * env,
            sin(u_fy * s + u_py) * env
        ) * u_scale;
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, u_thick * 0.3, best);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "fx":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
              "description": "x frequency"},
    "fy":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
              "description": "y frequency"},
    "px":    {"glsl": "float", "min": 0.0, "max": 6.28, "default": 0.0,
              "description": "x phase"},
    "py":    {"glsl": "float", "min": 0.0, "max": 6.28, "default": 1.57,
              "description": "y phase"},
    "decay": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.2,
              "description": "amplitude decay"},
    "turns": {"glsl": "float", "min": 1.0, "max": 12.0, "default": 6.0,
              "description": "number of turns"},
    "steps": {"glsl": "int", "min": 60, "max": 400, "default": 300,
              "description": "trace resolution"},
    "scale": {"glsl": "float", "min": 0.3, "max": 0.95, "default": 0.8,
              "description": "figure size"},
    "thick": {"glsl": "float", "min": 0.01, "max": 0.12, "default": 0.04,
              "description": "trace thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "drift speed"},
    "bg":    {"glsl": "color", "default": "#05070e", "description": "background"},
    "fg":    {"glsl": "color", "default": "#7ad7ff", "description": "trace color"},
})