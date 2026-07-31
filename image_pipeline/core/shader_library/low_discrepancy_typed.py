"""low_discrepancy_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("low_discrepancy_typed",
          "Low-discrepancy (R2) point field: stipple / dot pattern (typed, node 433)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.15 * u_speed;
    vec3 col = u_bg;
    // R2 low-discrepancy sequence (Roberts 2018): alpha = (1/phi^2, 1/phi^3).
    vec2 alpha = vec2(0.7548776662, 0.5698402909);
    int N = int(u_count);
    float best = 1e9;
    // Rasterise N dots; highlight the single nearest dot per pixel.
    for (int i = 0; i < 20000; i++) {
        if (i >= N) break;
        float fi = float(i);
        vec2 q = fract(alpha * fi + vec2(u_ox, u_oy) + t * 0.05);
        q -= 0.5; q.x *= u_resolution.x / u_resolution.y;
        // gentle rotation so animation is visible on the point cloud
        float ca = cos(t * 0.3), sa = sin(t * 0.3);
        q = mat2(ca, -sa, sa, ca) * q;
        best = min(best, length(p - q));
    }
    float dot = smoothstep(u_radius, u_radius * 0.3, best);
    col = mix(u_bg, inferno(clamp(1.0 - best * 1.5, 0.0, 1.0)), dot);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "count":   {"glsl": "int", "min": 50, "max": 20000, "default": 2000,
                "description": "number of sampled points N"},
    "radius":  {"glsl": "float", "min": 0.5, "max": 8.0, "default": 1.5,
                "description": "dot radius in px"},
    "ox":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "sequence x offset (seed)"},
    "oy":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                "description": "sequence y offset (seed)"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "rotation speed"},
    "bg":      {"glsl": "color", "default": "#05060c", "description": "background"},
})