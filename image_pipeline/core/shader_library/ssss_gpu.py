"""ssss_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 438 Subsurface Scatter / SSSS (client-GPU twin) ──
_register("ssss_gpu",
          "Subsurface Scatter SSSS (client-GPU twin of node 438)",
          "filter", _filter_typed('''
    // Separable exponential-profile blur (Jimenez & Gutierrez 2010): a sharp
    // core term + a broad halo term, two 1-D passes (X then Y). Strength
    // breathes with cos(u_time) so the preview is live.
    int N = int(u_samples);
    N = clamp(N, 4, 25);
    float ext = max(1e-3, u_radius * 3.0);
    float stp = ext / float(max(N, 1));
    float cs = max(1e-3, u_radius * max(0.1, u_falloff));
    float invCs = 1.0 / cs;
    float invCs2 = 1.0 / (cs * 2.5);
    float strength = clamp(u_strength * (0.7 + 0.3 * cos(u_time * u_anim_speed)), 0.0, 1.0);
    vec3 accx = orig.rgb; float wsumx = 1.0;
    for (int i = 0; i < 25; i++) {
        if (i >= N) break;
        float off = (float(i) + 0.5) * stp;
        float w = exp(-off * invCs) + 0.5 * exp(-off * invCs2);
        vec2 d = vec2(off, 0.0) / u_resolution;
        accx += w * (sample(clamp(uv + d, 0.0, 1.0)).rgb + sample(clamp(uv - d, 0.0, 1.0)).rgb);
        wsumx += 2.0 * w;
    }
    accx /= wsumx;
    vec3 accy = accx; float wsumy = 1.0;
    for (int i = 0; i < 25; i++) {
        if (i >= N) break;
        float off = (float(i) + 0.5) * stp;
        float w = exp(-off * invCs) + 0.5 * exp(-off * invCs2);
        vec2 d = vec2(0.0, off) / u_resolution;
        accy += w * (sample(clamp(uv + d, 0.0, 1.0)).rgb + sample(clamp(uv - d, 0.0, 1.0)).rgb);
        wsumy += 2.0 * w;
    }
    accy /= wsumy;
    vec3 outc = mix(orig.rgb, accy, strength);
    f_color = vec4(outc, 1.0);
'''), uniforms={
    "radius":     {"glsl": "float", "min": 2.0, "max": 60.0, "default": 18.0,
                   "description": "scatter radius in px"},
    "samples":    {"glsl": "float", "min": 4.0, "max": 25.0, "default": 11.0,
                   "description": "profile samples per axis"},
    "falloff":    {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.4,
                   "description": "profile sharpness"},
    "strength":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.85,
                   "description": "subsurface blend amount"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0,
                   "description": "animation speed"},
})