"""marble_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




_register("marble_typed", "Marble — Perlin-turbulence veining with domain warp (typed, node 320)",
          "procedural", '''float _turb(vec2 p, int oct) {
    // Absolute-value fbm (Perlin turbulence): sharp filaments instead of soft fbm.
    float sum = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= oct) break;
        sum  += amp * abs(noise(p * freq) * 2.0 - 1.0);
        norm += amp;
        amp  *= 0.5;
        freq *= 2.0;
    }
    return sum / max(norm, 0.001);
}
void main() {
    vec2 uv = v_uv;
    vec2 p = uv;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_speed * 0.1;

    // Slowly drifting turbulence field warps the vein coordinate.
    vec2 warp = vec2(t, -t * 0.6);
    float turb = _turb(p * u_scale + warp, int(u_octaves + 0.5));

    // Directional vein axis (angle in turns) + turbulence distortion.
    float ang = u_angle * 6.2831853;
    float axis = p.x * cos(ang) + p.y * sin(ang);
    float veins = axis * u_freq + turb * u_distortion;

    // Sharp sinusoidal bands -> marble veins.
    float m = 0.5 + 0.5 * sin(veins * 6.2831853);
    m = pow(m, max(u_sharpness, 0.01));

    vec3 col = mix(u_base_color, u_vein_color, clamp(m, 0.0, 1.0));

    // Subtle self-shadow from the raw turbulence for depth.
    col *= 0.75 + 0.25 * (1.0 - turb);

    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "speed":      {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "turbulence drift speed"},
    "scale":      {"glsl": "float", "min": 0.5, "max": 12.0, "default": 3.0,
                   "description": "turbulence field scale"},
    "octaves":    {"glsl": "int", "min": 1.0, "max": 6.0, "default": 5.0,
                   "description": "turbulence octaves (detail)"},
    "freq":       {"glsl": "float", "min": 0.5, "max": 16.0, "default": 4.0,
                   "description": "vein frequency"},
    "angle":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.12,
                   "description": "vein direction (turns)"},
    "distortion": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 2.0,
                   "description": "turbulence distortion of veins"},
    "sharpness":  {"glsl": "float", "min": 0.2, "max": 8.0, "default": 2.5,
                   "description": "vein contrast/sharpness"},
    "base_color": {"glsl": "color", "default": "#1a1c22", "description": "stone base colour"},
    "vein_color": {"glsl": "color", "default": "#e8e4d8", "description": "vein colour"},
})