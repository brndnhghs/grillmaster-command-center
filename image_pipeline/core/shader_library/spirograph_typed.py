"""spirograph_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed-uniform pattern expansions (ids 265–270, 2026-07-11) ───────────────
# Closed-form f(uv,t) pattern generators — every frame a pure function of the
# fragment coordinate and the animation clock, so the CPU-vs-GPU live preview is
# EXACTLY reproducible (no seeded-layout divergence). Additive GPU live path;
# no CPU fn is touched. Each declares NAMED typed uniforms that the factory
# turns into node params + wireable SCALAR ports.

_register("spirograph_typed", "Hypotrochoid/epitrochoid spirograph ribbons (typed, node 265)",
          "procedural", '''void main() {
    float R = max(u_ring_radius, 0.01);
    float r = max(u_wheel_radius, 0.001);
    float d = u_pen_offset;
    int np = int(clamp(u_petals, 1.0, 60.0));
    float speed = u_time * 0.02 * max(u_spin, 0.0);
    vec2 ctr = (vec2(u_center_x, u_center_y) - 0.5) * 2.0;
    vec2 p = (v_uv - 0.5) * 2.0 - ctr;
    float best = 1e9;
    for (int i = 0; i < 60; i++) {
        if (i >= np) break;
        float a = (float(i) / float(np)) * 6.2831853 + speed;
        float ca = cos(a), sa = sin(a);
        // hypotrochoid point for this phase
        vec2 q = (R - r) * vec2(ca, sa) + d * vec2(cos(((R - r) / r) * a), sin(((R - r) / r) * a));
        best = min(best, distance(p, q));
    }
    float rib = smoothstep(u_line_width, u_line_width * 0.25, best);
    f_color = vec4(mix(u_bg, u_ink, rib), 1.0);
}
''', uniforms={
    "ring_radius":  {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.7,
                     "description": "fixed ring radius R"},
    "wheel_radius": {"glsl": "float", "min": 0.02, "max": 0.9, "default": 0.27,
                     "description": "rolling wheel radius r"},
    "pen_offset":   {"glsl": "float", "min": 0.0, "max": 0.9, "default": 0.45,
                     "description": "pen distance from wheel center"},
    "petals":       {"glsl": "float", "min": 1.0, "max": 60.0, "default": 24.0,
                     "description": "number of lobes"},
    "spin":         {"glsl": "float", "min": 0.0, "max": 8.0, "default": 1.0,
                     "description": "rotation animation speed"},
    "center_x":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                     "description": "rosette center x"},
    "center_y":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                     "description": "rosette center y"},
    "line_width":   {"glsl": "float", "min": 0.005, "max": 0.08, "default": 0.02,
                     "description": "ribbon thickness"},
    "bg":   {"glsl": "color", "default": "#0c0e1a", "description": "background"},
    "ink":  {"glsl": "color", "default": "#5ef2c0", "description": "ribbon color"},
})

# ── Typed math_art: Spirograph (node 500) + Flowing Truchet (node 531) ──
# Both are closed-form f(uv,t) Architecture-B generators (no inter-frame
# state), so they are P0 (not P1). CPU fns stay authoritative for export;
# these are additive typed-uniform live-preview twins. Colormodes that are
# not closed-form (palette/inferno/viridis for 531; the full colormap set for
# 500) are dropped (GPU_PREVIEW_DROP_ALLOW) — the twins render a sensible
# default (IQ rainbow / inferno) and animate continuously from u_time.
_register("spirograph_typed", "Spirograph rosette (hypotrochoid/epitrochoid) — typed twin of node 500",
          "procedural", '''void main() {
    vec2 uv = v_uv - 0.5;
    uv.x *= u_resolution.x / u_resolution.y;
    float R = max(u_R, u_r + 1.0);
    float r = max(u_r, 1.0);
    float d = u_d;
    bool epi = false;  // hypotrochoid default; mode switch is CPU-authoritative (GPU_PREVIEW_DROP_ALLOW)
    float span = (epi ? (R + r) : (R - r)) + d + 1e-6;
    float k = (epi ? (R + r) : (R - r)) / r;
    // Generous closure range; bounded sample loop keeps it a per-pixel SDF.
    float theta_max = 6.2831853 * 24.0;
    const int STEPS = 1200;
    float dmin = 1e9;
    float bestHue = 0.0;
    for (int i = 0; i < STEPS; i++) {
        float th = (float(i) / float(STEPS)) * theta_max;
        vec2 p;
        if (epi) {
            p = vec2((R + r) * cos(th) - d * cos(k * th),
                     (R + r) * sin(th) - d * sin(k * th));
        } else {
            p = vec2((R - r) * cos(th) + d * cos(k * th),
                     (R - r) * sin(th) - d * sin(k * th));
        }
        // Continuous animation from u_time: rotate + breathe (no cusps).
        float ang = u_time;
        float ca = cos(ang), sa = sin(ang);
        p = vec2(ca * p.x - sa * p.y, sa * p.x + ca * p.y);
        float s = 0.4 + 0.6 * (0.5 + 0.5 * sin(u_time * 0.5));
        p *= s;
        vec2 cuv = p * (0.46 / span);
        float dd = distance(uv, cuv);
        if (dd < dmin) { dmin = dd; bestHue = atan(p.y, p.x); }
    }
    // Stroke width bridges the sample spacing so the rosette reads solid.
    float spacing_uv = 0.46 * theta_max / float(STEPS);
    float lw = max(max(u_line_width, 0.5) * 0.004, spacing_uv * 0.6);
    float line = smoothstep(lw, lw * 0.4, dmin);
    // IQ cosine palette (rainbow) with hue rotation.
    float hue = fract(u_hue_shift + bestHue / 6.2831853 + 0.5);
    vec3 col = clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    vec3 bg = vec3(0.5);  // BG_DEFAULT neutral grey
    f_color = vec4(mix(bg, col, line), 1.0);
}
''', uniforms={
    "R":         {"glsl": "float", "min": 3.0, "max": 60.0, "default": 35.0,
                   "description": "fixed (big) circle radius"},
    "r":         {"glsl": "float", "min": 1.0, "max": 60.0, "default": 13.0,
                   "description": "rolling (small) circle radius"},
    "d":         {"glsl": "float", "min": 1.0, "max": 60.0, "default": 25.0,
                   "description": "pen offset from small-circle centre"},
    "line_width": {"glsl": "float", "min": 0.5, "max": 4.0, "default": 1.6,
                   "description": "stroke thickness"},
    "hue_shift":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "hue rotation of the colour ramp"},
})