"""domain_coloring_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── GPU-First categorical coverage: recent CPU nodes tagged gpu-twin-candidate ──
# 431 Domain Coloring / 433 Low-Discrepancy Field. Closed-form f(uv,t) twins so
# the recent CPU nodes get a client-GPU live-preview mirror. CPU numpy node stays
# authoritative for export (two-tier precision). NAMED typed uniforms equal the
# CPU node's real numeric params (contract #5).

_register("domain_coloring_typed",
          "Domain coloring of complex functions: phase portrait + contour grid (typed, node 431)",
          "procedural", '''void main() {
    // Complex plane: uv in [-scale, scale] around (center_x, center_y).
    vec2 uv = (v_uv - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;
    vec2 z = uv * u_scale + vec2(u_center_x, u_center_y);
    // Animate via the live-preview clock u_time (the client advances it so the
    // live preview moves). Rotate the plane while gently drifting the center —
    // mirrors the CPU node's rotate/drift anim modes without a dead phase param.
    float a = u_time * 0.4;
    float ca = cos(a), sa = sin(a);
    z = mat2(ca, -sa, sa, ca) * z;
    z += vec2(sin(u_time * 0.3), cos(u_time * 0.22)) * u_scale * 0.12;
    // f(z) = z^n (the node default 'poly' with exponent n == z_n family).
    float n = max(u_exponent, 2.0);
    float r = length(z), th = atan(z.y, z.x);
    vec2 f = pow(r, n) * vec2(cos(n * th), sin(n * th));
    // Phase portrait: hue = arg f / 2pi; lightness = (2/pi) atan|f|.
    float arg = atan(f.y, f.x) / 6.2831853 + 0.5;
    float mag = atan(length(f)) * 2.0 / 3.14159265;
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (arg + vec3(0.0, 0.333, 0.667)));
    // 'enhanced'/'grid' contour: darken on log|f| & phase lattice lines.
    float gl = abs(fract(log(length(f) + 1e-3) * 3.0) - 0.5);
    float lp = abs(fract(arg * 12.0) - 0.5);
    float grid = smoothstep(0.02, 0.12, min(gl, lp));
    col *= mix(1.0, grid, u_grid);
    col *= mag;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "exponent":  {"glsl": "float", "min": 2.0, "max": 12.0, "default": 3.0,
                  "description": "power n for z^n"},
    "scale":     {"glsl": "float", "min": 0.5, "max": 8.0, "default": 3.0,
                  "description": "view half-extent in the complex plane"},
    "center_x":  {"glsl": "float", "min": -4.0, "max": 4.0, "default": 0.0,
                  "description": "real part of view center"},
    "center_y":  {"glsl": "float", "min": -4.0, "max": 4.0, "default": 0.0,
                  "description": "imaginary part of view center"},
    "grid":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                  "description": "contour/grid overlay strength (artistic knob; CPU 'coloring' is a string mode, not a float synonym)"},
})