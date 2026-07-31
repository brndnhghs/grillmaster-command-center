"""stained_glass_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# 317 — Voronoi stained glass: F1/F2 cellular decomposition with a flat random
# facet color per cell and dark leaded seams along cell boundaries. Site jitter
# animates the cells so seams drift.
_register("stained_glass_typed", "Voronoi stained-glass facets with leaded seams (typed, node 317)",
          "procedural", '''void main() {
    vec2 p = v_uv;
    p.x *= u_resolution.x / u_resolution.y;
    float N = max(u_cells, 2.0);
    vec2 g = p * N;
    vec2 id = floor(g);
    vec2 f = fract(g);
    float d1 = 8.0, d2 = 8.0;
    vec2 bestId = id;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 rnd = vec2(hash21(id + o + 0.5), hash21(id + o + 31.4));
            rnd = 0.5 + u_jitter * 0.5 * sin(u_time * u_speed + 6.2831 * rnd);
            vec2 pt = o + rnd - f;
            float d = length(pt);
            if (d < d1) { d2 = d1; d1 = d; bestId = id + o; }
            else if (d < d2) { d2 = d; }
        }
    }
    vec3 facet = 0.35 + 0.65 * vec3(hash21(bestId + 2.1),
                                    hash21(bestId + 5.3),
                                    hash21(bestId + 9.7));
    float lum = dot(facet, vec3(0.333));
    facet = mix(vec3(lum), facet, u_saturation);
    float seam = smoothstep(0.0, max(u_seam, 1e-3), d2 - d1);
    vec3 col = mix(u_seam_color, facet, seam);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cells":      {"glsl": "float", "min": 2.0, "max": 30.0, "default": 9.0, "description": "cells per axis"},
    "jitter":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "site drift amount"},
    "seam":       {"glsl": "float", "min": 0.01, "max": 0.25, "default": 0.07, "description": "leaded seam width"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.85, "description": "facet color saturation"},
    "speed":      {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "cell animation speed"},
    "seam_color": {"glsl": "color", "default": "#0a0a0f", "description": "seam (lead) color"},
})