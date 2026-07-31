"""droste_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Categorical coverage pt.15 (2026-07-12): closed-form procedural patterns
# with NAMED typed controls — Droste log-spiral, Voronoi stained glass, Op-Art
# sinusoidal band distortion. Each is a pure f(uv,t) field (no ping-pong state)
# so it verifies headlessly via render_shader. ──

# 316 — Droste log-spiral: conformal log-polar mapping (Escher "Print Gallery"
# homage). Rings tile self-similarly in log-radius while the angle winds them
# into a spiral; animation rotates the spiral phase.
_register("droste_typed", "Droste log-polar self-similar spiral (typed, node 316)",
          "procedural", '''void main() {
    vec2 p = v_uv - 0.5;
    p.x *= u_resolution.x / u_resolution.y;
    float r = length(p);
    float a = atan(p.y, p.x);
    float lr = log(max(r, 1e-4));
    // Spiral coordinate: log-radius zoom + angular winding + time phase.
    float coord = lr * u_zoom + a * u_twist + u_time * u_speed * 0.5;
    float band = fract(coord * u_bands);
    float ring = smoothstep(0.46, 0.5, abs(band - 0.5));
    // Radial vignette so the singular centre fades cleanly.
    float vig = smoothstep(0.02, 0.15, r);
    vec3 col = mix(u_bg, u_fg, ring * vig);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "zoom":   {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.5, "description": "log-radius zoom (ring density)"},
    "twist":  {"glsl": "float", "min": 0.0, "max": 12.0, "default": 3.0, "description": "angular winding (spiral arms)"},
    "bands":  {"glsl": "float", "min": 1.0, "max": 16.0, "default": 4.0, "description": "band repetition"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.8, "description": "spiral animation speed"},
    "fg":     {"glsl": "color", "default": "#f0d060", "description": "ring color"},
    "bg":     {"glsl": "color", "default": "#101020", "description": "background color"},
})