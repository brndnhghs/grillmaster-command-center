"""flowing_truchet_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("flowing_truchet_typed", "Flowing Truchet labyrinth (domain-warped flow field) — typed twin of node 531",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = v_uv;
    float cells = max(u_scale, 1.0);
    vec2 g = uv * cells;
    vec2 id = floor(g);
    vec2 f = fract(g) - 0.5;
    // Domain-warped flow field (port of CPU _flow_angle). Fixed phase keeps
    // neighbouring tiles coherent into rivulet-like channels.
    vec2 fc = (id + 0.5) / cells * 6.2831853;
    float s1 = 0.5, s2 = 0.3;
    float wx = fc.x + u_warp * sin(fc.y * 1.7 + u_time * 0.5 + s1);
    float wy = fc.y + u_warp * cos(fc.x * 1.3 - u_time * 0.4 + s2);
    float a = sin(wx * 2.0 + u_time * 0.6) + cos(wy * 2.3 - u_time * 0.3);
    a += 0.5 * sin((wx + wy) * 3.1 + u_time * 0.8 + s1);
    bool bit = mod(a, 6.2831853) < 3.14159265;
    // Truchet arc SDF: two opposite-corner quarter arcs per cell.
    float rr = 0.5;
    float d1, d2;
    if (bit) {
        d1 = abs(distance(f, vec2(-0.5, -0.5)) - rr);
        d2 = abs(distance(f, vec2( 0.5,  0.5)) - rr);
    } else {
        d1 = abs(distance(f, vec2( 0.5, -0.5)) - rr);
        d2 = abs(distance(f, vec2(-0.5,  0.5)) - rr);
    }
    float d = min(d1, d2);
    float lw = max(u_line_width, 1.0) * 0.01;
    float line = smoothstep(lw, lw * 0.4, d);
    f_color = vec4(inferno(clamp(line, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 4.0, "max": 80.0, "default": 28.0,
                   "description": "tiles across the shorter canvas axis"},
    "line_width": {"glsl": "float", "min": 1.0, "max": 14.0, "default": 4.0,
                   "description": "arc stroke thickness (px)"},
    "warp":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "domain-warp strength (0=stripes, 1=rivulets)"},
})