"""sdf_scene_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── 950 SDF Scene (client-GPU twin) ──
_register("sdf_scene_gpu",
          "SDF Scene (client-GPU twin of node 950)",
          "procedural",
'''float sd_circle(vec2 p, float r) { return length(p) - r; }
float sd_box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
float sd_ring(vec2 p, float r, float th) { return abs(length(p) - r) - th; }
float smin_p(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 2.0 * u_scale;
    float t = u_time * u_anim_speed;

    // rotate + drift (closed-form, live). cos/sin drift, angle=t.
    vec2 p = uv + 0.30 * vec2(sin(t), cos(t * 0.7));
    p = rot(t * 0.5) * p;

    // domain repetition (tiling)
    if (u_repetition > 1e-4) {
        float rep = max(1e-3, u_repetition);
        p = mod(p + 0.5 * rep, rep) - 0.5 * rep;
    }
    float k = max(1e-3, u_blend);
    float dc = sd_circle(p, 0.16);
    float db = sd_box(p, vec2(0.20));
    float dr = sd_ring(p, 0.34, 0.022);
    float d = smin_p(smin_p(dc, db, k), dr, k);

    // shading from the field
    vec3 bg = vec3(0.03, 0.02, 0.05);
    vec3 ink = vec3(0.98, 0.78, 0.36);
    float edge = 0.014;
    float inside = clamp(0.5 - d / (2.0 * edge), 0.0, 1.0);
    float glow_eff = u_glow * (1.0 + 0.5 * cos(t));
    float glow_f = exp(-3.0 * max(d, 0.0)) * glow_eff;
    float band_f = 0.5 + 0.5 * sin(d * max(0.0, u_bands) - t);
    float band_factor = (1.0 - u_band_mix) + u_band_mix * band_f;

    vec3 col = mix(bg, ink, inside) + ink * glow_f;
    col *= band_factor;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "scale":      {"glsl": "float", "min": 0.5, "max": 4.0, "default": 1.6, "description": "scene zoom / world scale"},
    "blend":      {"glsl": "float", "min": 0.01, "max": 0.6, "default": 0.18, "description": "smooth-min blend softness"},
    "repetition": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.0, "description": "domain-repetition cell size (0=off)"},
    "glow":       {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "outside halo strength"},
    "bands":      {"glsl": "float", "min": 0.0, "max": 40.0, "default": 12.0, "description": "distance isoline count"},
    "band_mix":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "contour-band modulation amount"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
})