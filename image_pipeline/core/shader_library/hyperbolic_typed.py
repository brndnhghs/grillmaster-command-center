"""hyperbolic_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# 307 — Poincaré-disk hyperbolic {p,q} tiling via repeated inversion in the
# edge-circles of the central regular p-gon. Edges glow; interior fills with
# inferno by inversion depth.
_register("hyperbolic_typed", "Poincaré-disk hyperbolic {p,q} tiling (typed, node 307)",
          "procedural", _INFERNO_GPU + '''void main() {
    int p = int(clamp(u_sides, 3.0, 12.0));
    int q = int(clamp(u_verts, 3.0, 12.0));
    float cp = cos(3.14159265 / float(p));
    float cq = cos(3.14159265 / float(q));
    float R0 = cq / cp;
    float dm = (R0 * R0 + 1.0) / (2.0 * R0 * cp);
    float rho = sqrt(max(dm * dm - 1.0, 1e-4));
    vec2 uv = (v_uv - 0.5) * 2.0;
    float t = u_time * u_speed;
    uv = rot(t * 0.25) * uv;
    vec3 col = u_bg;
    if (length(uv) < 1.0) {
        vec2 pnt = uv;
        float it = 0.0;
        for (int i = 0; i < 6; i++) {
            float best = 1e9;
            vec2 bc = vec2(0.0);
            for (int j = 0; j < 12; j++) {
                if (j >= p) break;
                float th = (float(j) + 0.5) * 6.2831853 / float(p);
                vec2 c = vec2(cos(th), sin(th)) * dm;
                float d2 = dot(pnt - c, pnt - c);
                if (d2 < best) { best = d2; bc = c; }
            }
            vec2 d = pnt - bc;
            float d2 = max(dot(d, d), 1e-6);
            pnt = bc + (rho * rho / d2) * d;
            it += 1.0;
        }
        col = mix(u_bg, inferno(clamp(0.2 + 0.6 * fract(it * 0.25), 0.0, 1.0)), 0.85);
        col = mix(col, u_edge, smoothstep(0.05, 0.0, abs(length(pnt) - rho)));
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "sides": {"glsl": "float", "min": 3.0, "max": 12.0, "default": 5.0, "description": "p — polygon sides"},
    "verts": {"glsl": "float", "min": 3.0, "max": 12.0, "default": 4.0, "description": "q — polygons at a vertex"},
    "speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.4, "description": "rotation speed"},
    "bg":    {"glsl": "color", "default": "#05060f", "description": "background"},
    "edge":  {"glsl": "color", "default": "#39e0ff", "description": "edge glow color"},
})