"""fourier_circles_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("fourier_circles_typed", "Fourier epicycles: traced harmonic curve (typed, node 274)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float best = 1e9;
    for (int i = 0; i < 128; i++) {
        float s = float(i) / 127.0;
        float ph = s * 6.28318530 + t;
        vec2 q = vec2(0.0);
        q.x += sin(ph * u_freq1 + u_phase1) * (0.32 / max(u_freq1, 1.0));
        q.y += cos(ph * u_freq2 + u_phase2) * (0.32 / max(u_freq2, 1.0));
        q += 0.22 * vec2(sin(ph * u_freq3), cos(ph * u_freq3));
        best = min(best, length(p - q));
    }
    float line = smoothstep(u_thick, 0.0, best);
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq1":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
               "description": "harmonic 1 freq"},
    "freq2":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 5.0,
               "description": "harmonic 2 freq"},
    "freq3":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 2.0,
               "description": "harmonic 3 freq"},
    "phase1": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0,
               "description": "harmonic 1 phase"},
    "phase2": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 1.2,
               "description": "harmonic 2 phase"},
    "thick":  {"glsl": "float", "min": 0.005, "max": 0.08, "default": 0.02,
               "description": "line thickness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "animation speed"},
    "bg":     {"glsl": "color", "default": "#05070f", "description": "background"},
    "fg":     {"glsl": "color", "default": "#62f0c8", "description": "curve color"},
})