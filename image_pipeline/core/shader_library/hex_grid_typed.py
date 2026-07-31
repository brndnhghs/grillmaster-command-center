"""hex_grid_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("hex_grid_typed", "Hexagonal lattice with tri-planar cell tinting (typed, node 268)",
          "procedural", '''vec4 _hexDist(vec2 p) {
    // p in hex-tile space (unit cell). Returns vec4(r, g, b, minDist) where
    // the first three components carry the three edge distances of the hexagon.
    vec2 q = abs(p);
    float c = dot(q, normalize(vec2(1.0, 1.7320508)));
    float a = max(c, q.x);
    float b = max(c, q.y);
    // distance to the two relevant edge orientations + vertical edge
    vec2 r = vec2(max(a, b), max(q.x * 0.8660254 + q.y * 0.5, q.y));
    return vec4(a, b, q.y, min(a, r.y));
}
void main() {
    float sc = max(u_scale, 0.5);
    vec2 p = (v_uv - 0.5) * u_resolution / sc * 2.0;
    p += vec2(u_offset_x, u_offset_y) * u_resolution / sc * 2.0;
    p.y += u_time * u_flow * 0.5;
    const vec2 s = vec2(1.0, 1.7320508);
    vec2 a = mod(p, s) - s * 0.5;
    vec2 b = mod(p + s * 0.5, s) - s * 0.5;
    vec4 ha = _hexDist(a);
    vec4 hb = _hexDist(b);
    float d = (length(a) < length(b)) ? ha.w : hb.w;
    // thickness is in CELL units (0..0.5); convert so 0.1 reads as a thin wall.
    float th = max(u_thickness * 0.1, 0.001);
    float edge = smoothstep(th, th * 0.4, d);
    vec2 cell = (length(a) < length(b)) ? floor(p / s) : floor((p + s * 0.5) / s);
    float idh = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
    vec3 fill = mix(u_fill_a, u_fill_b, idh);
    f_color = vec4(mix(fill, u_line, edge), 1.0);
}
''', uniforms={
    "scale":       {"glsl": "float", "min": 4.0, "max": 120.0, "default": 24.0,
                    "description": "hex cell size (px)"},
    "thickness":   {"glsl": "float", "min": 0.5, "max": 8.0, "default": 2.0,
                    "description": "wall thickness (px)"},
    "flow":        {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "downward drift speed"},
    "offset_x":    {"glsl": "float", "min": -0.5, "max": 0.5, "default": 0.0,
                    "description": "horizontal offset"},
    "offset_y":    {"glsl": "float", "min": -0.5, "max": 0.5, "default": 0.0,
                    "description": "vertical offset"},
    "fill_a":      {"glsl": "color", "default": "#14233f", "description": "cell tint A"},
    "fill_b":      {"glsl": "color", "default": "#2a4d6e", "description": "cell tint B"},
    "line":        {"glsl": "color", "default": "#9fe3ff", "description": "wall color"},
})