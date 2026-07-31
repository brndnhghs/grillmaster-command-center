"""moire_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("moire_typed", "Moiré interference gratings with typed mode/speed/freq (node 250)",
          "procedural", '''void main() {
    int mode = int(clamp(floor(u_mode + 0.5), 0.0, 3.0));
    float s1 = max(u_speed1, 0.05);
    float s2 = max(u_speed2, 0.05);
    float freq = max(u_freq, 1.0);
    float t = u_time * 0.05;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float scale = 1.0 / max(res.x, res.y) * 2.0 * 3.14159265;
    float a1 = s1 * t, a2 = s2 * t;
    float g1, g2;
    if (mode == 1) {            // linear
        g1 = 0.5 + 0.5 * sin(freq * (p.x * cos(a1) + p.y * sin(a1)) * scale);
        g2 = 0.5 + 0.5 * sin(freq * (p.x * cos(a2) + p.y * sin(a2)) * scale);
    } else if (mode == 2) {     // spiral
        float r = length(p);
        g1 = 0.5 + 0.5 * sin(freq * r * scale + a1 * 4.0);
        g2 = 0.5 + 0.5 * sin(freq * r * scale + a2 * 4.0);
    } else if (mode == 3) {     // hex
        vec2 h = vec2(p.x, abs(fract(p.y * 0.5) - 0.25)) * scale;
        g1 = 0.5 + 0.5 * sin(freq * (h.x + a1));
        g2 = 0.5 + 0.5 * sin(freq * (h.y + a2));
    } else {                    // radial
        float r = length(p);
        g1 = 0.5 + 0.5 * sin(freq * r * scale + a1 * 4.0);
        g2 = 0.5 + 0.5 * sin(freq * r * scale + a2 * 4.0 + 1.57);
    }
    float v = clamp(0.5 + 0.5 * sin((g1 - g2) * 3.14159), 0.0, 1.0);
    f_color = vec4(mix(u_color_a, u_color_b, v), 1.0);
}
''', uniforms={
    "mode":   {"glsl": "choice", "choices": ["radial", "linear", "spiral", "hex"],
               "default": "radial", "description": "interference geometry"},
    "speed1": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
               "description": "grating 1 speed"},
    "speed2": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.28,
               "description": "grating 2 speed"},
    "freq":   {"glsl": "float", "min": 1.0, "max": 60.0, "default": 20.0,
               "description": "grating frequency"},
    "color_a": {"glsl": "color", "default": "#0b1026", "description": "low color"},
    "color_b": {"glsl": "color", "default": "#ffcf4d", "description": "high color"},
})