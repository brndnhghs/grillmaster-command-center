"""guilloche_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("guilloche_typed", "Guilloché: rose-curve engraving lattice (typed, node 278)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.06 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    float rose = cos(a * u_petals + t) * u_amp;
    float bands = sin((r - rose) * u_freq * 6.28318530);
    float line = smoothstep(1.0 - u_sharp, 1.0, abs(bands));
    vec3 col = mix(u_bg, u_ink, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "petals": {"glsl": "float", "min": 2.0, "max": 24.0, "default": 7.0,
               "description": "rose petal count"},
    "amp":    {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.08,
               "description": "rose amplitude"},
    "freq":   {"glsl": "float", "min": 4.0, "max": 60.0, "default": 24.0,
               "description": "ring frequency"},
    "sharp":  {"glsl": "float", "min": 0.05, "max": 0.9, "default": 0.4,
               "description": "line sharpness"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "animation speed"},
    "bg":     {"glsl": "color", "default": "#060a10", "description": "background"},
    "ink":    {"glsl": "color", "default": "#7cf0ff", "description": "engraving"},
})