"""foam_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("foam_typed", "Procedural bubble foam / Voronoi cell membrane (typed, node 300)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 uv = v_uv * u_cells;
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    vec2 g = floor(uv); vec2 f = fract(uv);
    float md = 1e9; vec2 mp = vec2(0.0);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 o = vec2(float(i), float(j));
            vec2 cell = g + o;
            vec2 off = vec2(hash21(cell), hash21(cell + 3.3));
            off = 0.5 + 0.45 * sin(t + 6.2831 * off);
            vec2 r = o + off - f;
            float dd = dot(r, r);
            if (dd < md) { md = dd; mp = r; }
        }
    }
    float dist = sqrt(md);
    float edge = smoothstep(u_thick, 0.0, dist);
    float irid = fract(dist * u_irid + u_hue_shift);
    vec3 col = mix(u_bg, _hsv(irid, u_sat, 1.0), edge);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "cell jitter speed"},
    "cells":    {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0,
                "description": "cell count"},
    "thick":    {"glsl": "float", "min": 0.01, "max": 0.4, "default": 0.12,
                "description": "membrane thickness"},
    "irid":     {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.5,
                "description": "iridescence bands"},
    "hue_shift":{"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.55,
                "description": "hue offset"},
    "sat":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.7,
                "description": "saturation"},
    "bg":       {"glsl": "color", "default": "#06121a", "description": "background"},
})