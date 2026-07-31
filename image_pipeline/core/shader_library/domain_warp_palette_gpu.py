"""domain_warp_palette_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




_register("domain_warp_palette_gpu",
          "Domain warping — fbm(fbm(p + fbm(p))) feedback (Inigo Quilez, 2015) with "
          "a 4-colour palette; two-level noise feed-forward produces marbled, organic "
          "flow; animated by scrolling the inner warp offset with u_time",
          "procedural", '''float _warpfbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) {
        v += a * noise(p); p *= 2.02; a *= 0.5;
    }
    return v;
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_speed;
    // Two-level domain warp: q = fbm(p + t), r = fbm(p + q + t), final = fbm(p + r).
    vec2 p = uv * u_scale + u_offset;
    vec2 q = vec2(_warpfbm(p + vec2(0.0, 0.0) + 0.15 * t),
                  _warpfbm(p + vec2(5.2, 1.3) - 0.10 * t));
    vec2 r = vec2(_warpfbm(p + u_warp * q + vec2(1.7, 9.2) + 0.12 * t),
                  _warpfbm(p + u_warp * q + vec2(8.3, 2.8) - 0.08 * t));
    float f = _warpfbm(p + u_warp * r);
    // Mix palette by the three warp layers for the classic iq marble look.
    vec3 col = mix(u_color_a, u_color_b, clamp(f * f * 2.0, 0.0, 1.0));
    col = mix(col, u_color_c, clamp(length(q), 0.0, 1.0));
    col = mix(col, u_color_d, clamp(r.x, 0.0, 1.0));
    col = u_color_bg + col * (0.25 + 0.6 * f);
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":   {"glsl": "float", "min": 0.5, "max": 8.0, "default": 3.0,
                  "description": "pattern frequency (zoom into the warp field)"},
    "warp":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.6,
                  "description": "warp strength (feed-forward coupling)"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                  "description": "animation scroll speed"},
    "offset":  {"glsl": "float", "min": 0.0, "max": 20.0, "default": 0.0,
                  "description": "static field offset into the warp field"},
    "color_a": {"glsl": "color", "default": "#1a2a6c",
                  "description": "base palette colour A"},
    "color_b": {"glsl": "color", "default": "#b21f1f",
                  "description": "palette colour B (f-weighted)"},
    "color_c": {"glsl": "color", "default": "#fdbb2d",
                  "description": "palette colour C (q-weighted)"},
    "color_d": {"glsl": "color", "default": "#16a085",
                  "description": "palette colour D (r.x-weighted)"},
    "color_bg": {"glsl": "color", "default": "#04060f",
                  "description": "background tint"},
})