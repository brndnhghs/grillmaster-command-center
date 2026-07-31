"""opart_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# 318 — Op-Art band distortion: parallel bands whose position is sinusoidally
# displaced (Bridget Riley homage). The whole field rotates and the wave phase
# animates for a shimmering moiré effect.
_register("opart_typed", "Op-Art sinusoidal band distortion (typed, node 318)",
          "procedural", '''void main() {
    vec2 p = v_uv - 0.5;
    p.x *= u_resolution.x / u_resolution.y;
    p = rot(u_rotation) * p;
    float disp = u_amplitude * sin(p.x * u_freq_x * 6.2831 + u_time * u_speed);
    float y = p.y + disp;
    float band = fract(y * u_bands);
    float stripe = smoothstep(0.47, 0.5, abs(band - 0.5));
    vec3 col = mix(u_bg, u_fg, stripe);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "bands":     {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0, "description": "band frequency"},
    "amplitude": {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.12, "description": "wave displacement"},
    "freq_x":    {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0, "description": "horizontal wave frequency"},
    "rotation":  {"glsl": "float", "min": 0.0, "max": 3.14159, "default": 0.0, "description": "field rotation (radians)"},
    "speed":     {"glsl": "float", "min": 0.0, "max": 5.0, "default": 1.2, "description": "wave animation speed"},
    "fg":        {"glsl": "color", "default": "#f5f5f5", "description": "stripe color"},
    "bg":        {"glsl": "color", "default": "#101014", "description": "background color"},
})