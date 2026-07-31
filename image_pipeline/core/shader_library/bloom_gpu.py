"""bloom_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("bloom_gpu", "Soft additive bloom / glow on the input (typed)",
          "filter", '''
vec3 _bloom_sample(vec2 uv, float r) {
    vec3 s = vec3(0.0);
    for (int k = 0; k < 8; k++) {
        float a = float(k) / 7.0 * 6.28318530;
        s += texture(u_texture, uv + vec2(cos(a), sin(a)) * r).rgb;
    }
    return s / 8.0;
}
void main() {
    vec2 uv = v_uv;
    vec3 src = texture(u_texture, uv).rgb;
    float r = u_radius * 0.03;
    // Two-pass cheap bloom (wide + tight) for a soft halo.
    vec3 glow = _bloom_sample(uv, r) * 0.6 + _bloom_sample(uv, r * 0.4) * 0.4;
    glow = pow(glow, vec3(max(u_threshold, 0.01)));   // emphasize bright areas
    vec3 col = mix(src, src + glow * u_strength * 1.6, clamp(u_strength, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                 "description": "glow strength"},
    "radius":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4,
                 "description": "glow radius"},
    "threshold":{"glsl": "float", "min": 0.1, "max": 2.0, "default": 1.0,
                 "description": "brightness threshold (gamma)"},
})