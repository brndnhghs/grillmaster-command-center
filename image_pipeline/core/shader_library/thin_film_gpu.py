"""thin_film_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("thin_film_gpu", "Thin-film interference iridescence (client-GPU twin of node 419)",
          "filter", '''
void main() {
    // Radial thickness field (matches the CPU node's default 'radial' source):
    // d grows from the frame center outward, so the iridescent bands form a
    // soap-bubble / oil-slick ring pattern over the wired substrate.
    vec2 p = v_uv - 0.5;
    float r = length(p) * 1.4;
    // Live-preview animation: advance the radial thickness with the preview
    // clock u_time so the iridescent bands drift (mirrors the CPU node's
    // anim_mode/time). The client feeds u_time every frame.
    float d = u_thickness + u_thickness_range * (r + 0.06 * sin(u_time * 0.6));
    float ang = radians(u_angle);
    float sin_a = sin(ang);
    float c = sin_a / max(u_ior, 1.001);
    float cos_t = sqrt(max(0.0, 1.0 - c * c));
    // Optical path difference (nm); per-wavelength reflectance via R(λ)=cos².
    float opd = 2.0 * u_ior * d * cos_t;
    vec3 lam = vec3(650.0, 550.0, 450.0);
    vec3 phase = (6.2831853 * opd / lam) + 3.14159265;
    vec3 iri = (1.0 - cos(phase)) * 0.5;
    // Saturation control around the band luminance.
    float lum = dot(iri, vec3(0.3333333));
    iri = clamp(lum + u_saturation * (iri - lum), 0.0, 1.0);
    vec3 src = texture(u_texture, v_uv).rgb;
    vec3 col = mix(src, iri, u_strength);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "thickness":        {"glsl": "float", "min": 100.0, "max": 1200.0, "default": 380.0,
                        "description": "base film thickness (nm)"},
    "thickness_range":  {"glsl": "float", "min": 0.0, "max": 1200.0, "default": 320.0,
                        "description": "thickness variation (nm)"},
    "ior":              {"glsl": "float", "min": 1.0, "max": 2.5, "default": 1.33,
                        "description": "film refractive index"},
    "angle":            {"glsl": "float", "min": 0.0, "max": 80.0, "default": 0.0,
                        "description": "incidence angle (deg)"},
    "strength":         {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                        "description": "overlay blend over source"},
    "saturation":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 1.0,
                        "description": "band color saturation"},
})