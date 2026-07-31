"""hash_field_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register







_register("hash_field_gpu",
          "Multiresolution Hash Encoding field (client-GPU twin of node 309)",
          "procedural", '''vec3 hash3(vec2 p) {
    vec3 q = vec3(dot(p, vec2(127.1, 311.7)),
                   dot(p, vec2(269.5, 183.3)),
                   dot(p, vec2(419.2, 371.9)));
    return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

// Bilinearly-interpolated integer-lattice hash (shared table == NGP look).
float hfeat(vec2 g, float lvl) {
    vec2 i = floor(g);
    vec2 f = g - i;
    vec2 u = f * f * (3.0 - 2.0 * f);          // smoothstep
    // cheap hash into [-1,1] (use a different per-level salt)
    vec2 h00 = hash3(i + vec2(0.0, 0.0) + lvl * 7.0).xy;
    vec2 h10 = hash3(i + vec2(1.0, 0.0) + lvl * 7.0).xy;
    vec2 h01 = hash3(i + vec2(0.0, 1.0) + lvl * 7.0).xy;
    vec2 h11 = hash3(i + vec2(1.0, 1.0) + lvl * 7.0).xy;
    float top = mix(h00.x, h10.x, u.x);
    float bot = mix(h01.x, h11.x, u.x);
    return mix(top, bot, u.y);
}

void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,16] 0.5 -> 4 ; p2 detail(levels) [1,16] 0.5 -> 10
    //   p3 hue  [0,1]    0.5 -> 0.5 ; p4 contrast [0.2,2.5] 0.5 -> 1.35
    float scale  = 1.0 + u_scale * 15.0;
    float levels = clamp(1.0 + u_detail * 15.0, 1.0, 16.0);
    float hue    = u_hue;
    float contrast = 0.2 + u_contrast * 2.3;

    vec2 uv = v_uv;
    float tt = u_time;
    // smooth coordinate animation (matches CPU anim_mode=pan/zoom/swirl/pulse)
    uv = fract(uv + vec2(tt * 0.05, tt * 0.03));
    float s = exp(-tt * 0.15);
    uv = (uv - 0.5) * s + 0.5;

    float acc = 0.0;
    for (int l = 0; l < 16; l++) {
        if (float(l) >= levels) break;
        float N = max(1.0, scale * pow(2.0, float(l)));
        acc += hfeat(uv * N, float(l));
    }
    acc /= levels;
    float val = clamp(0.5 + 0.5 * acc * contrast, 0.0, 1.0);
    // IQ cosine palette with hue shift
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (val + vec3(0.0, 0.33, 0.67)) + hue * 6.2831853);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 1.0, "max": 16.0, "default": 4.0, "description": "coarsest grid resolution"},
    "detail": {"glsl": "float", "min": 1.0, "max": 16.0, "default": 10.0, "description": "hash-grid levels (octaves)"},
    "hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette hue shift"},
    "contrast": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.35, "description": "tone contrast"},
})