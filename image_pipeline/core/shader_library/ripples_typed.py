"""ripples_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ripples_typed", "Concentric color ripples with typed freq/speed/colors (node 257)",
          "procedural", '''void main() {
    vec2 uv = v_uv - 0.5;
    float d = length(uv);
    float ph = u_time * max(u_speed, 0.1) - d * max(u_freq, 1.0);
    float r = 0.5 + 0.5 * sin(ph);
    float g = 0.5 + 0.5 * sin(ph + 2.0);
    float b = 0.5 + 0.5 * sin(ph + 4.0);
    vec3 col = mix(u_color_a, u_color_b, d);
    col *= vec3(r, g, b);
    col *= (1.0 - d);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":  {"glsl": "float", "min": 1.0, "max": 40.0, "default": 30.0,
              "description": "ripple frequency"},
    "speed": {"glsl": "float", "min": 0.1, "max": 4.0, "default": 2.0,
              "description": "ripple speed"},
    "color_a": {"glsl": "color", "default": "#10071f", "description": "inner color"},
    "color_b": {"glsl": "color", "default": "#4affd0", "description": "outer color"},
})