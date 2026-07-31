"""gabor_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



















# ── Gabor Noise (CPU node 477 twin) — anisotropic sparse Gabor convolution ──
# Lagae et al. 2011. One Gabor kernel per jittered-lattice cell; sum the
# kernels in each pixel's neighbourhood. p1=orientation(deg), p2=anisotropy,
# p3=frequency, p4=bandwidth. Gentle u_time rotation for the live preview.
_register("gabor_gpu", "Gabor noise (Lagae et al. 2011) — anisotropic sparse Gabor convolution", "procedural", '''
void main() {
    vec2 res = u_resolution;
    vec2 px = v_uv * res;
    float orient = radians(90.0);   // fixed: node 473 exposes no orientation param
    float aniso = clamp(u_anisotropy, 0.0, 1.0);
    float freq = clamp(u_frequency * 12.0, 0.5, 12.0);
    float bw = clamp(u_falloff * 6.0, 0.5, 6.0);

    float S = clamp(70.0 / freq, 6.0, 90.0);
    float Fmag = freq * 0.03;
    float bwf = clamp(bw / 2.5, 0.4, 2.4);
    float spar = S * 0.34 * bwf;
    float sper = spar * max(0.12, 1.0 - aniso * 0.85);

    // gentle time rotation (live-preview animation)
    vec2 u = vec2(cos(orient + u_time * 0.5), sin(orient + u_time * 0.5));

    float acc = 0.0;
    float R = 3.0;
    for (float o = 0.0; o < 2.0; o += 1.0) {
        float So = S / pow(2.0, o);
        float Fm = Fmag * pow(2.0, o);
        float spo = spar / pow(2.0, o) * bwf;
        float spe = sper / pow(2.0, o);
        float amp = 1.0 / pow(2.0, o);
        float ci = floor(px.x / So);
        float cj = floor(px.y / So);
        for (float dj = -R; dj <= R; dj += 1.0) {
            for (float di = -R; di <= R; di += 1.0) {
                float nci = ci + di;
                float ncj = cj + dj;
                float jx = (hash21(vec2(nci, ncj)) - 0.5) * So;
                float jy = (hash21(vec2(nci + 131.0, ncj + 517.0)) - 0.5) * So;
                vec2 ip = vec2(nci * So + jx, ncj * So + jy);
                float ph = hash21(vec2(nci + 977.0, ncj + 331.0)) * 6.2831853;
                vec2 d = px - ip;
                float dp = dot(d, u);
                float dq = dot(d, vec2(-u.y, u.x));
                float env = exp(-3.14159265 * (dp * dp / (spo * spo) + dq * dq / (spe * spe)));
                acc += amp * env * cos(6.2831853 * Fm * dp + ph);
            }
        }
    }
    float v = clamp(acc / (abs(acc) + 0.6), -1.0, 1.0);
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (0.5 + 0.5 * v) + vec3(0.0, 0.33, 0.67));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "anisotropy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "kernel elongation (0=isotropic, 1=fully stretched)"},
    "frequency":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "spatial frequency (scaled x12 internally)"},
    "falloff":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "envelope bandwidth (scaled x6 internally)"},
})