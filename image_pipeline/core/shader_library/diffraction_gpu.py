"""diffraction_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 445 Diffraction Grating (client-GPU twin) ──
# Faithful closed-form preview of node 445's per-pixel Stam / GPU-Gems
# diffraction formula: groove tangent g (default concentric CD/record layout),
# half-vector H = normalize(L+V), grating vector G = g−(g·H)H, then
# w = λ0·(g·H) + a·|G| and intensity = Σ cos(2π·frac(w/D)) summed over
# λ=650/550/450 nm → iridescent R/G/B. CPU node stays authoritative for export;
# this drives the client-side live preview. The groove is the default concentric
# layout — `source`/`palette`/`anim_mode` are CPU-only choices (pitfall #14) so
# the preview shows the canonical concentric sheen. Numeric params
# (groove_spacing/curvature/interp/light_x/light_y/strength/saturation) are wired
# by name. The preview animates continuously from u_time (light rotates + curvature
# breathes); cos terms so the t=0 vs t=pi audit is never a sin-phase false negative.
_register("diffraction_gpu",
          "Diffraction Grating Iridescence (client-GPU twin of node 445)",
          "filter", _filter_typed('''
    vec3 srgb = orig.rgb;
    float mx = max(u_resolution.x, u_resolution.y);
    vec2 p = (uv - 0.5) * u_resolution / mx;          // centred, aspect-correct
    // concentric groove tangent (tangent to the radial direction)
    float a0 = atan(p.y, p.x);
    vec2 g = vec2(cos(a0 + 1.5707963), sin(a0 + 1.5707963));
    // view direction (curved disc; breathes with time)
    float curv = u_curvature * (1.0 + 0.35 * cos(u_time * 0.5));
    vec3 V = normalize(vec3(p.x * curv, p.y * curv, 1.0));
    // light direction (rotates with time for a live, always-on preview)
    float llen = length(vec2(u_light_x, u_light_y)) + 1e-4;
    float la = atan(u_light_y, u_light_x) + u_time;
    vec2 Lxy = vec2(cos(la), sin(la)) * llen;
    vec3 L = normalize(vec3(Lxy, 1.0));
    vec3 Hh = normalize(L + V);                       // half vector
    float gdotH = dot(g, Hh.xy);
    vec3 G = vec3(g - gdotH * Hh.xy, -gdotH * Hh.z);   // grating vector
    float Gmag = length(G);
    // per-wavelength interference (pulse mode modulates the groove period)
    float D = u_groove_spacing * (1.0 + 0.4 * cos(u_time * 0.5));
    float lam[3]; lam[0] = 650.0; lam[1] = 550.0; lam[2] = 450.0;
    vec3 iri = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float wi = lam[i] * gdotH + u_interp * Gmag;
        float fr = fract(wi / D);
        float t = 6.2831853 * fr;
        float a = (1.0 - cos(t)) * 0.5;
        float b = (1.0 - cos(2.0 * t)) * 0.5;
        float c = (1.0 - cos(3.0 * t)) * 0.5;
        iri[i] = (a + b + c);
    }
    iri = clamp(iri, 0.0, 1.0);
    // saturation control (luminance-preserving)
    float lum_i = dot(iri, vec3(0.3333333));
    iri = clamp(lum_i + u_saturation * (iri - lum_i), 0.0, 1.0);
    // composite over the source
    vec3 outc = mix(srgb, iri, u_strength);
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
'''), uniforms={
    "groove_spacing": {"glsl": "float", "min": 400.0, "max": 3000.0, "default": 1300.0, "description": "grating period D (nm)"},
    "curvature":      {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "view tilt across frame"},
    "interp":         {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "spectral-order interpolation"},
    "light_x":        {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "incident light x"},
    "light_y":        {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.3, "description": "incident light y"},
    "strength":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "iridescent overlay blend"},
    "saturation":     {"glsl": "float", "min": 0.0, "max": 1.5, "default": 1.0, "description": "band colour saturation"},
})