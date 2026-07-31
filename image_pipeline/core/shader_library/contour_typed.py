"""contour_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("contour_typed", "Topographic contour map of FBM terrain with typed levels/thickness (node 292)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float n = fbm(p * u_freq + u_time * 0.03 * u_speed);
    vec3 col = mix(u_color_a, u_color_b, clamp(n, 0.0, 1.0));
    float c = n * u_levels;
    float d = abs(fract(c) - 0.5) * 2.0;
    float line = smoothstep(u_thick, u_thick * 0.3, d);
    col = mix(col, u_line, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
               "description": "terrain drift"},
    "freq":   {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.5,
               "description": "terrain feature size"},
    "levels": {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0,
               "description": "contour line count"},
    "thick":  {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.12,
               "description": "line thickness"},
    "color_a": {"glsl": "color", "default": "#0d3b2e", "description": "low elevation"},
    "color_b": {"glsl": "color", "default": "#e8d8a0", "description": "high elevation"},
    "line":   {"glsl": "color", "default": "#1a1208", "description": "contour ink"},
})