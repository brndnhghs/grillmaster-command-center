"""burst_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("burst_typed", "Radial energy burst / shockwave (typed, node 299)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    float wave = sin(r * u_freq - t * u_velocity) * 0.5 + 0.5;
    float spokes = pow(abs(cos(a * u_spokes * 0.5 + t * 0.2)), u_sharpness);
    float ring = smoothstep(u_thick, 0.0, abs(r - fract(t * u_velocity * 0.05) * u_reach)) * u_intensity;
    float energy = (wave * 0.4 + spokes * 0.4 + ring * 0.8);
    energy *= smoothstep(u_reach, 0.0, r);
    vec3 col = mix(u_bg, u_hot, clamp(energy, 0.0, 1.0));
    col = mix(col, u_core, smoothstep(0.6, 1.0, energy));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":      {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "animation speed"},
    "freq":       {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0,
                "description": "radial wave frequency"},
    "velocity":   {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0,
                "description": "shockwave speed"},
    "spokes":     {"glsl": "float", "min": 1.0, "max": 24.0, "default": 8.0,
                "description": "spoke count"},
    "sharpness":  {"glsl": "float", "min": 1.0, "max": 12.0, "default": 4.0,
                "description": "spoke sharpness"},
    "thick":      {"glsl": "float", "min": 0.01, "max": 0.3, "default": 0.06,
                "description": "ring thickness"},
    "reach":      {"glsl": "float", "min": 0.4, "max": 2.0, "default": 1.4,
                "description": "burst reach"},
    "intensity":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8,
                "description": "ring intensity"},
    "bg":         {"glsl": "color", "default": "#05060f", "description": "background"},
    "hot":        {"glsl": "color", "default": "#ff7a18", "description": "hot ring"},
    "core":       {"glsl": "color", "default": "#fff2c4", "description": "core flash"},
})