"""gerstner_ocean_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register





# -- Categorical coverage: Gerstner Ocean (typed GPU node 352) --
# Closed-form trochoidal-wave ocean (Fournier & Reeves 1986; GPU Gems ch.1
# normal, Finch 2004) as a shaded height field with Blinn-Phong sun glitter.
# Mirrors the CPU node 963 (patterns/gerstner_ocean.py) which explicitly flags
# "a clean GPU twin is a natural follow-up". Genuinely time-varying (wave phases
# advance with u_time) so it survives the contrast-only static cull.
# CPU numpy node stays authoritative for export (two-tier precision).
_register("gerstner_ocean_gpu",
          "Gerstner Ocean -- analytic trochoidal wave height field with sun "
          "glitter (typed GPU twin of node 963); genuinely animated (wave "
          "phases advance with u_time)",
          "procedural", """
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.6;
    float N = clamp(u_n_waves, 1.0, 24.0);
    float baseL = max(u_base_wavelength, 1e-3);
    float falloff = clamp(u_wavelength_falloff, 0.5, 0.95);
    float amp = u_amplitude;
    float Q = clamp(u_steepness, 0.0, 1.0);
    float windA = u_wind_angle * 6.2831853;
    float spread = u_wind_spread * 6.2831853;

    float z = 0.0;
    vec2 slope = vec2(0.0);
    float nz = 1.0;
    float L = baseL;
    float A = amp;
    float sumA = 0.0;
    for (float i = 0.0; i < 24.0; i += 1.0) {
        if (i >= N) break;
        float frac = (N > 1.0) ? (i / (N - 1.0) - 0.5) : 0.0;
        float ang = windA + frac * spread + hash21(vec2(i, 3.7)) * 0.4;
        vec2 D = vec2(cos(ang), sin(ang));
        float k = 6.2831853 / L;
        float w = sqrt(9.8 * k * 0.02);
        float phi = hash21(vec2(i, 11.3)) * 6.2831853;
        float f = k * dot(D, uv) - w * t + phi;
        z += A * sin(f);
        slope += D * (k * A * cos(f));
        nz -= Q * k * A * sin(f);
        sumA += A;
        L *= falloff;
        A *= falloff;
    }
    if (sumA > 0.0) z /= sumA;

    vec3 Nn = normalize(vec3(-slope, max(nz, 0.05)));
    float sunAz = u_sun_angle * 6.2831853;
    float sunEl = clamp(u_sun_height, 0.05, 1.0) * 1.5707963;
    vec3 sun = normalize(vec3(cos(sunAz) * cos(sunEl),
                              sin(sunAz) * cos(sunEl), sin(sunEl)));
    vec3 viewv = vec3(0.0, 0.0, 1.0);
    vec3 hv = normalize(sun + viewv);
    float diff = max(dot(Nn, sun), 0.0);
    float spec = pow(max(dot(Nn, hv), 0.0), max(u_shininess, 1.0)) * u_glint;

    float crest = clamp(z * 3.0 + 0.5, 0.0, 1.0);
    vec3 deep = 0.5 + 0.5 * cos(6.2831853 * (u_deep_hue + vec3(0.0, 0.15, 0.3)));
    deep *= vec3(0.25, 0.5, 0.7);
    vec3 crestc = 0.5 + 0.5 * cos(6.2831853 * (u_crest_hue + vec3(0.0, 0.1, 0.2)));
    crestc = mix(crestc, vec3(0.9, 0.95, 1.0), 0.5);
    vec3 col = mix(deep, crestc, crest);
    col = col * (0.25 + 0.9 * diff) + vec3(spec);
    col *= u_exposure;
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 1e-3)));
    f_color = vec4(col, 1.0);
}
""", uniforms={
    "n_waves":            {"glsl": "float", "min": 1.0, "max": 24.0, "default": 9.0, "description": "number of Gerstner wave components"},
    "base_wavelength":    {"glsl": "float", "min": 0.15, "max": 2.5, "default": 1.1, "description": "longest wavelength (screen units)"},
    "wavelength_falloff": {"glsl": "float", "min": 0.5, "max": 0.95, "default": 0.82, "description": "per-wave wavelength scale (<1 shortens)"},
    "amplitude":          {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.16, "description": "master wave amplitude"},
    "steepness":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.7, "description": "trochoid steepness Q (0=round, 1=sharp crests)"},
    "wind_angle":         {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.12, "description": "dominant wind direction (turns)"},
    "wind_spread":        {"glsl": "float", "min": 0.0, "max": 0.4, "default": 0.14, "description": "angular spread of wave directions (turns)"},
    "sun_angle":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.62, "description": "sun azimuth (turns)"},
    "sun_height":         {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.35, "description": "sun elevation (0=horizon, 1=zenith)"},
    "shininess":          {"glsl": "float", "min": 4.0, "max": 400.0, "default": 90.0, "description": "specular sharpness of the sun glitter"},
    "glint":              {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.6, "description": "sun-glitter intensity"},
    "deep_hue":           {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.53, "description": "deep-water base hue"},
    "crest_hue":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "crest/foam hue"},
    "exposure":           {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.1, "description": "overall brightness multiplier"},
    "gamma":              {"glsl": "float", "min": 0.3, "max": 2.5, "default": 1.0, "description": "tonal gamma"},
})