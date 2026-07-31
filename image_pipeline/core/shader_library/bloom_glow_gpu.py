"""bloom_glow_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed filter twins for CPU filter nodes (categorical GPU coverage) ──
# Each is a closed-form per-pixel approximation of the CPU node it shadows;
# the CPU numpy node stays authoritative for export (two-tier precision).
# Uniform names equal the CPU node's real numeric params (contract #5) so the
# browser live preview tracks the sliders. Choice/string params (source,
# palette, anim_mode, mode, paper, ink, aperture_shape) are intentionally
# unmapped (pitfall #14) — the twin filters whatever image is wired in.
_register("bloom_glow_gpu", "Bloom / glow with optional anamorphic streak (typed twin of node 408)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    float thr = u_threshold;
    float knee = max(thr * u_softness, 0.001);
    vec2 px = 1.0 / u_resolution;
    float r = max(u_radius, 1.0);
    vec3 glow = vec3(0.0);
    float wsum = 0.0;
    const int N = 16;
    float ga = 2.39996323;
    // Golden-angle disc sampling of the bright-pass = a cheap single-pass glow.
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float rad = sqrt((fi + 0.5) / float(N)) * r;
        float ang = fi * ga;
        vec2 off = vec2(cos(ang) * u_streak, sin(ang)) * rad;
        vec3 s = texture(u_texture, v_uv + off * px).rgb;
        float l = dot(s, vec3(0.2126, 0.7152, 0.0722));
        float f = clamp((l - (thr - knee)) / (2.0 * knee), 0.0, 1.0);
        f = f * f;
        glow += s * f;
        wsum += max(f, 0.0001);
    }
    glow /= max(wsum, 0.001);
    vec3 col = src + u_intensity * glow;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "threshold": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "brightness cutoff for the bloom prefilter"},
    "softness":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "soft-knee width as fraction of threshold"},
    "intensity": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.2,
                  "description": "glow additive strength"},
    "radius":    {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0,
                  "description": "blur radius in px (glow spread)"},
    "streak":    {"glsl": "float", "min": 1.0, "max": 8.0, "default": 1.0,
                  "description": "anamorphic streak anisotropy (1=round)"},
})