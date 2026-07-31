"""checker_gpu2 — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("checker_gpu2", "Checkerboard with typed tile counts, colors, rotation",
          "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = radians(u_angle);
    uv = rot(a) * uv + 0.5;
    vec2 tiles = vec2(max(u_tiles_x, 1.0), max(u_tiles_y, 1.0));
    vec2 cellPos = fract(uv * tiles);
    float chk = mod(floor(uv.x * tiles.x) + floor(uv.y * tiles.y), 2.0);
    vec3 col = mix(u_color_a, u_color_b, chk);
    // Optional grid lines between tiles.
    if (u_line_width > 0.001) {
        vec2 edge = min(cellPos, 1.0 - cellPos);
        float line = step(min(edge.x, edge.y), u_line_width * 0.5);
        col = mix(col, u_line_color, line);
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "tiles_x":    {"glsl": "float", "min": 1.0, "max": 64.0, "default": 8.0,
                   "description": "tiles across"},
    "tiles_y":    {"glsl": "float", "min": 1.0, "max": 64.0, "default": 8.0,
                   "description": "tiles down"},
    "angle":      {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                   "description": "rotation (deg)"},
    "color_a":    {"glsl": "color", "default": "#101018", "description": "tile color A"},
    "color_b":    {"glsl": "color", "default": "#e8e4d8", "description": "tile color B"},
    "line_width": {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.0,
                   "description": "grid line width (0 = none)"},
    "line_color": {"glsl": "color", "default": "#4a9eff", "description": "grid line color"},
})