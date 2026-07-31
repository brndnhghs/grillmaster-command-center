"""truchet_maze_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("truchet_maze_typed", "Random-rotated Truchet arc/maze tiling (typed, node 266)",
          "procedural", '''float _truchet_hash(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
void main() {
    int cells = int(clamp(u_cells, 1.0, 40.0));
    float cellSize = 1.0 / float(cells);
    // Continuous rotation of the whole tiling with time -> every frame differs
    // and the maze appears to spin/re-tile as it animates.
    float ang = u_time * 0.15 * max(u_anim_speed, 0.0);
    mat2 R = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
    vec2 uv = R * (v_uv - 0.5) / cellSize + 0.5;
    vec2 id = floor(uv);
    vec2 f = fract(uv) - 0.5;
    float h = _truchet_hash(id);
    bool flip = h > 0.5;
    if (flip) f = vec2(f.y, f.x);
    float r = u_arc_radius;
    float d1 = abs(distance(f, vec2(-0.5 + r, -0.5 + r)) - r);
    float d2 = abs(distance(f, vec2( 0.5 - r,  0.5 - r)) - r);
    float d = min(d1, d2);
    float line = smoothstep(u_line_width, u_line_width * 0.4, d);
    vec3 col = mix(u_bg, u_ink, line);
    if (u_show_nodes > 0.5) {
        float dn = min(distance(f, vec2(-0.5, -0.5)), distance(f, vec2(0.5, 0.5)));
        col = mix(col, u_node_color, smoothstep(0.06, 0.02, dn));
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cells":       {"glsl": "float", "min": 1.0, "max": 40.0, "default": 8.0,
                    "description": "tiles per axis"},
    "line_width":  {"glsl": "float", "min": 0.01, "max": 0.25, "default": 0.09,
                    "description": "stroke width"},
    "arc_radius":  {"glsl": "float", "min": 0.1, "max": 0.5, "default": 0.5,
                    "description": "arc radius (1=semicircle)"},
    "anim_speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "tile rotation animation speed"},
    "show_nodes":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "draw connection nodes (0/1)"},
    "bg":          {"glsl": "color", "default": "#10131f", "description": "background"},
    "ink":         {"glsl": "color", "default": "#e8d9a0", "description": "stroke color"},
    "node_color":  {"glsl": "color", "default": "#ff6b6b", "description": "node color"},
})