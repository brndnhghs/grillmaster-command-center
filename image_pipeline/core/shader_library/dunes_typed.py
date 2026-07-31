"""dunes_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dunes_typed", "Sand dune migration with typed wind/sediment (node 252)",
          "procedural", '''void main() {
    float wind = max(u_wind_strength, 0.0);
    float sed = max(u_sediment, 0.0);
    float t = u_time * 0.05;
    float windAngle = t * 0.15;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float h = 0.0;
    // Layered wave superposition -> migrating dune field.
    // wind_strength scales the wave amplitude (stronger wind -> taller, higher-contrast
    // dunes); sediment controls wavelength (more sediment -> finer ripples).
    float amp_scale = 0.2 + 1.5 * clamp(wind, 0.0, 1.5);   // height grows with wind
    float wlen_base = mix(60.0, 8.0, clamp(sed, 0.0, 1.0)); // finesse with sediment
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float ang = windAngle + fi * 0.7;
        vec2 dir = vec2(cos(ang), sin(ang));
        float wlen = wlen_base * (1.0 - fi / 8.0);
        float amp = amp_scale * (1.0 - fi / 6.0);
        h += amp * sin(dot(p, dir) / wlen + t * (1.0 + fi * 0.2));
    }
    // Fixed normalization (independent of wind) so wind_strength changes contrast.
    float v = clamp(0.5 + 0.5 * (h / 5.0), 0.0, 1.0);
    vec3 col = mix(u_sand_low, u_sand_high, v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "wind_strength": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                      "description": "wind strength (dune height)"},
    "sediment":      {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.8,
                      "description": "sediment supply (ripple fineness)"},
    "sand_low":  {"glsl": "color", "default": "#5a3a1a", "description": "shadow sand"},
    "sand_high": {"glsl": "color", "default": "#e8c89a", "description": "lit sand"},
})