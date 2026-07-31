"""domainwarp_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed-uniform closed-form pattern batch (2026-07-11, nodes 295-300) ──
# Extended family of single-output procedural nodes with NAMED typed controls.
# Each is a pure function of (uv, t) — exact parity live preview, no seeded
# layout divergence. Reuses the prologue helpers (fbm/noise/hash21/rot) and an
# inlined hsv2rgb (no dependency on the late _INFERNO_GPU helper).

_register("domainwarp_typed", "Domain-warped fractal flow field (typed, node 295)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    vec2 q = vec2(fbm(p * u_scale + t), fbm(p * u_scale + 5.2 - t));
    vec2 r = vec2(fbm(p * u_scale + u_warp * q + 1.7),
                  fbm(p * u_scale + u_warp * q + 9.2));
    float v = fbm(p * u_scale + u_warp * r);
    float hue = fract(v * u_hue_spread + u_hue_shift);
    vec3 col = _hsv(hue, u_sat, 0.35 + 0.65 * v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":        {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                    "description": "flow speed"},
    "scale":        {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.5,
                    "description": "noise frequency"},
    "warp":         {"glsl": "float", "min": 0.0, "max": 6.0, "default": 3.0,
                    "description": "domain-warp iterations"},
    "hue_shift":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                    "description": "base hue"},
    "hue_spread":   {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.7,
                    "description": "hue range across field"},
    "sat":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8,
                    "description": "saturation"},
})