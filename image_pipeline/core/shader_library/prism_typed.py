"""prism_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("prism_typed", "Spectral prism / diffraction grating (typed, node 297)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.03 * u_speed;
    float d = dot(p, vec2(cos(u_angle), sin(u_angle)));
    float spec = sin(d * u_freq + t) * 0.5 + 0.5;
    float bands = spec * u_rainbow;
    vec3 col = _hsv(fract(bands + u_hue_shift), u_sat, 1.0);
    float vig = smoothstep(u_falloff, 0.0, length(p));
    col *= mix(1.0, vig, u_darken);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":     {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "phase drift"},
    "freq":      {"glsl": "float", "min": 2.0, "max": 60.0, "default": 22.0,
                "description": "grating frequency"},
    "angle":     {"glsl": "float", "min": 0.0, "max": 360.0, "default": 30.0,
                "description": "grating angle (deg)"},
    "rainbow":   {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.2,
                "description": "hue spread"},
    "hue_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "hue offset"},
    "sat":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.9,
                "description": "saturation"},
    "falloff":   {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
                "description": "edge falloff"},
    "darken":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                "description": "edge darkening"},
})