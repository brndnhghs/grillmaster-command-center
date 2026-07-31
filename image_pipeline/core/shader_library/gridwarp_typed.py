"""gridwarp_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("gridwarp_typed", "Domain-warped grid lattice with typed warp/cells/width (node 294)",
          "procedural", '''void main() {
    vec2 g = v_uv * u_cells;
    vec2 w = vec2(
        fbm(g * 0.5 + u_time * 0.05 * u_speed),
        fbm(g * 0.5 + 7.3 - u_time * 0.05 * u_speed)
    ) - 0.5;
    g += w * u_warp;
    vec2 f = fract(g);
    float lx = smoothstep(u_width, 0.0, min(f.x, 1.0 - f.x));
    float ly = smoothstep(u_width, 0.0, min(f.y, 1.0 - f.y));
    float line = max(lx, ly);
    vec3 col = mix(u_bg, u_line, line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
               "description": "warp flow speed"},
    "cells":  {"glsl": "float", "min": 2.0, "max": 60.0, "default": 14.0,
               "description": "grid cell count"},
    "width":  {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.12,
               "description": "line width"},
    "warp":   {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.7,
               "description": "domain warp strength"},
    "bg":     {"glsl": "color", "default": "#0a0a12", "description": "background"},
    "line":   {"glsl": "color", "default": "#43e8d8", "description": "grid line"},
})