"""dot_noise_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




_register("dot_noise_gpu", "Aperiodic gyroid dot-noise fBm (Xor, GM Shaders 2025)", "procedural", '''
// Dot Noise — a cheap closed-form alternative to 3D simplex noise.
//   Ref: Xor, "Dot Noise", GM Shaders Mini, 2025-09-05
//        https://mini.gmshaders.com/p/dot-noise
// Core idea: gyroid = dot(cos(p), sin(p.yzx)); giving one axis an
// irrational (golden-ratio) frequency makes the sheets never realign,
// yielding aperiodic pseudo-noise with NO hash lookups — ideal for
// many-sample volumetric-style sampling. Here it is fBm-summed and
// animated by sweeping the 3rd (z) coordinate through u_time.
//   u_params.x = base frequency / zoom   (0.5 -> 6.0)
//   u_params.y = fBm octaves            (0.5 -> 4)
//   u_params.z = warp amount (self-domain-warp)  (0.5 -> 0.35)
//   u_params.w = color palette phase    (0.5 -> 0.5)
// PHI = golden ratio -> "most irrational" frequency for aperiodicity.
float dotGyroid(vec3 p) {
    // aperiodic gyroid: one swizzled axis carries a phi-scaled frequency
    const float PHI = 1.61803398875;
    vec3 q = vec3(p.x, p.y * PHI, p.z);
    return dot(cos(q), sin(q.yzx));
}
float dotNoiseFbm(vec3 p, float oct, float warp) {
    // self domain-warp for richer structure (cheap: one extra eval)
    float w = dotGyroid(p * 0.6);
    p += warp * vec3(w, w * 0.7, w * 1.3);
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        if (float(i) >= oct) break;
        sum += amp * dotGyroid(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return sum;
}
void main() {
    vec2 uv = (v_uv * 2.0 - 1.0);
    uv.x *= u_resolution.x / u_resolution.y;

    float baseFreq = u_freq;
    float oct = floor(u_octaves + 0.5);
    float warp = u_warp;
    float phase = u_palette;

    // sweep the 3rd coordinate through time: animates the noise field
    vec3 p = vec3(uv * baseFreq, u_time * u_flow);

    float n = dotNoiseFbm(p, oct, warp);
    // gyroid dot is in ~[-2,2]; remap to [0,1]
    float v = clamp(n * 0.25 + 0.5, 0.0, 1.0);

    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (phase + v + vec3(0.0, 0.33, 0.67)));
    col *= 0.35 + 0.65 * v;
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":    {"glsl": "float", "min": 2.0, "max": 10.0, "default": 6.0,
                "description": "base frequency / zoom of the noise field"},
    "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 4.0,
                "description": "number of fBm octaves summed"},
    "warp":    {"glsl": "float", "min": 0.0, "max": 0.7, "default": 0.35,
                "description": "self domain-warp amount"},
    "palette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                "description": "cosine palette phase"},
    "flow":    {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.6,
                "description": "time-sweep speed through the noise volume"},
})