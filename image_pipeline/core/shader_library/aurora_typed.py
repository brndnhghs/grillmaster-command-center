"""aurora_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# 319 — Aurora Borealis: real-time procedural northern-lights curtain. A
# domain-warped sinusoidal energy field produces vertical "rays" that drift with
# time; a Gaussian sky-band window localises the curtain, and an x/y-driven hue
# ramp sweeps green→violet. Closed-form f(uv,t) — no texture, no raymarch loop —
# so it is a cheap procedural twin (good live-preview + fast export). References:
# Roy Theunissen's "Aurora Borealis: A Breakdown" (2022) for the layered-curtain
# model and the GodotShaders volumetric-aurora approach for the energy-field look.
_register("aurora_typed", "Aurora Borealis — real-time drifting light-curtain (typed, node 319)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_speed * 0.2;

    // Sky gradient background (dark zenith, faint blue at horizon).
    vec3 sky = mix(u_sky_bottom, u_sky_top, pow(uv.y, 0.6));

    // Aurora occupies an upper band of the sky.
    float band = exp(-pow((uv.y - u_center) / u_thickness, 2.0));

    // Warped horizontal field drives the curtain ribbons (vertical streaks).
    float x = p.x * u_scale;
    float warp = fbm(vec2(x * 0.4, t)) * 2.5;
    float ph = x + warp + t * 1.3;

    // Several drifting ribbon layers; rays fade upward so they rise like light.
    float ribbon = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float amp = 1.0 - fi * 0.25;
        float s = sin(ph * (1.0 + fi * 0.5) + fi * 2.0);
        float streak = smoothstep(0.75, 1.0, abs(s)) * amp;
        streak *= smoothstep(u_center + u_thickness, u_center - u_thickness, uv.y);
        ribbon += streak;
        wsum += amp;
    }
    ribbon /= max(wsum, 0.001);

    float rays = ribbon * band;

    // Colour: green base with violet tips, swept by x and height.
    float hue = mix(u_hue_green, u_hue_violet,
                    clamp(0.5 + 0.5 * sin(x * 0.15 + t), 0.0, 1.0));
    hue = fract(hue - uv.y * 0.15);
    vec3 aurora = _hsv(hue, 0.9, 1.0) * rays * 1.6;

    // Faint static star speckle outside the band.
    float stars = step(0.996, hash21(floor(uv * u_resolution / 2.0))) * (1.0 - band);
    sky += stars * 0.5;

    vec3 col = clamp(sky + aurora, 0.0, 1.0);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":      {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "curtain drift speed"},
    "scale":      {"glsl": "float", "min": 0.5, "max": 12.0, "default": 3.0,
                   "description": "ribbon frequency"},
    "center":     {"glsl": "float", "min": 0.2, "max": 0.9, "default": 0.62,
                   "description": "curtain height (sky band centre)"},
    "thickness":  {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.14,
                   "description": "curtain thickness"},
    "hue_green":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.33,
                   "description": "base green hue"},
    "hue_violet": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.78,
                   "description": "tip violet hue"},
    "sky_bottom": {"glsl": "color", "default": "#02030a", "description": "horizon sky"},
    "sky_top":    {"glsl": "color", "default": "#0a1430", "description": "zenith sky"},
})