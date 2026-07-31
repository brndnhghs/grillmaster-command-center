"""smin_metaballs_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _inferno_local



# ── Node 321: Smooth-min Metaballs (client-GPU twin, closed-form f(uv,t)) ──
# Research technique: Quilez exponential/quadratic smooth-minimum (smin) merging
# of signed-distance spheres → organically blended "metaball" surfaces. The
# existing node-53 metaballs uses a sum-of-inverse-square field (no true SDF /
# no smin); this twin implements the canonical smin union so the `blend` (k)
# param genuinely controls edge softness the way IQ describes it. Closed-form
# f(uv,t) → exact parity preview; CPU numpy stays authoritative for export.
_register("smin_metaballs_gpu",
          "Smooth-min Metaballs — IQ exponential smin union of orbiting SDF spheres (node 321)",
          "procedural", _inferno_local('') + '''
// Quilez smooth minimum (exponential variant, k = blend).
// Reference: https://iquilezles.org/articles/smin/
float smin_exp(float a, float b, float k) {
    k = max(k, 1e-4);
    float res = exp2(-a / k) + exp2(-b / k);
    return -k * log2(max(res, 1e-4));
}
// Signed distance to a moving ball indexed i.
float ball(vec2 p, float fi, float t, float r) {
    float a = fi * 2.39996323 + t * (0.6 + 0.05 * fi);   // golden-angle spread
    float orbit = 0.18 + 0.16 * hash21(vec2(fi, 1.7));
    vec2 c = vec2(0.5 + orbit * cos(a), 0.5 + orbit * sin(a * 1.3));
    return length(p - c) - r;
}
void main() {
    // 0.5-neutral encoding → node defaults (pitfall #15).
    int   nBalls = int(clamp(3.0 + u_count * 9.0, 3.0, 12.0)); // 3..12
    float k      = mix(0.004, 0.10, u_blend);                 // smin blend
    float thr    = mix(0.55, 0.02, u_threshold);             // edge threshold
    float hue    = u_hue;                                    // base hue 0..1
    float speed  = mix(0.15, 2.6, u_ball_speed);
    float t = u_time * 0.05 * speed;

    vec2 p = v_uv;
    float r = 0.055 + 0.045 * hash21(vec2(7.0, 3.0));
    float d = 1e5;
    for (int i = 0; i < 12; i++) {
        if (float(i) >= float(nBalls)) break;
        float di = ball(p, float(i), t, r);
        d = smin_exp(d, di, k);            // smooth-union the SDFs
    }
    // Normalize the signed field into a 0..1 surface band + interior fill.
    float surf = 1.0 - smoothstep(0.0, thr, abs(d));
    float fill = 1.0 - smoothstep(0.0, thr * 0.6, d);
    float edge = smoothstep(thr, 0.0, abs(d));       // glow at the membrane
    float val = clamp(fill * 0.85 + surf * 0.4 + edge * 0.6, 0.0, 1.0);

    float ang = atan(p.y - 0.5, p.x - 0.5);
    vec3 col = inferno(val);
    // Tint with the hue control so the blob membrane shifts color by angle.
    vec3 tint = 0.5 + 0.5 * cos(6.2831853 * (hue + ang / 6.2831853) + vec3(0.0, 2.0, 4.0));
    col = mix(col, col * (0.6 + 0.6 * tint), 0.45 * edge);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "blend": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "smin blend (k) — edge softness"},
    "count": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "number of metaballs (3..12)"},
    "threshold": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "edge threshold (lower = fatter blobs)"},
    "ball_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "orbit speed"},
    "hue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "membrane tint hue"}
    }
    )