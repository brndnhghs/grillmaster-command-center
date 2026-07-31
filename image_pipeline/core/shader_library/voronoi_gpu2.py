"""voronoi_gpu2 — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("voronoi_gpu2", "Voronoi/worley cellular cells with typed controls",
          "procedural", '''
vec2 _cell(vec2 g, float seed) {
    return g + 0.5 + 0.5 * vec2(
        sin(seed + 3.1 * g.x + 1.7 * g.y),
        cos(seed + 2.3 * g.x - 4.1 * g.y));
}
void main() {
    vec2 uv = v_uv * max(u_scale, 0.5);
    uv += u_time * u_drift * vec2(0.13, 0.07);
    float seed = u_seed * 6.2831;
    vec2 g = floor(uv), f = fract(uv);
    float d1 = 1e9, d2 = 1e9;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec2 off = vec2(float(i), float(j));
        vec2 c = _cell(g + off, seed);
        float d = length(c - f);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
    float t = (u_metric == 1) ? (d2 - d1) : d1;   // F2-F1 edges vs nearest
    t = clamp(t * 1.6, 0.0, 1.0);
    if (u_cells > 0.5) t = step(0.5, t);           // hard cell regions
    vec3 col = mix(u_color_a, u_color_b, t);
    if (u_edge > 0.001) {
        float e = smoothstep(0.0, u_edge, abs(d1 - 0.5 * u_scale * 0.04));
        col = mix(u_edge_color, col, e);
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":     {"glsl": "float", "min": 1.0, "max": 32.0, "default": 8.0,
                  "description": "cell density"},
    "seed":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "cell layout seed"},
    "drift":     {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.1,
                  "description": "animation drift speed"},
    "metric":    {"glsl": "choice", "choices": ["nearest", "edges"],
                  "default": "nearest", "description": "distance metric"},
    "cells":     {"glsl": "int", "min": 0, "max": 1, "default": 0,
                  "description": "hard cell regions (0=smooth)"},
    "edge":      {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.0,
                  "description": "cell boundary line width"},
    "color_a":   {"glsl": "color", "default": "#0a0a12", "description": "cell color A"},
    "color_b":   {"glsl": "color", "default": "#37e0c8", "description": "cell color B"},
    "edge_color":{"glsl": "color", "default": "#ffffff", "description": "boundary color"},
})