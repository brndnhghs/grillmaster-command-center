"""waveform_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("waveform_typed", "Waveform: summed sine oscillators (typed, node 275)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.08 * u_speed;
    float y = 0.0;
    y += sin(p.x * u_k1 * 6.28318530 + t);
    y += 0.5 * sin(p.x * u_k2 * 6.28318530 + t * 1.3);
    y += 0.3 * sin(p.x * u_k3 * 6.28318530 + t * 0.7);
    y *= u_amp * 0.25;
    float line = smoothstep(u_thick, 0.0, abs(p.y - y));
    vec3 col = mix(u_bg, u_fg, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "k1":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 3.0,
              "description": "osc 1 wavenumber"},
    "k2":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 6.0,
              "description": "osc 2 wavenumber"},
    "k3":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 9.0,
              "description": "osc 3 wavenumber"},
    "amp":   {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
              "description": "amplitude"},
    "thick": {"glsl": "float", "min": 0.005, "max": 0.08, "default": 0.02,
              "description": "trace thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "scroll speed"},
    "bg":    {"glsl": "color", "default": "#04060d", "description": "background"},
    "fg":    {"glsl": "color", "default": "#ff6bd6", "description": "trace color"},
})