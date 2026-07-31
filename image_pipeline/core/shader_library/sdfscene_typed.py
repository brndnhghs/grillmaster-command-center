"""sdfscene_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sdfscene_typed", "Minimal signed-distance scene (typed, node 298)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
float _sdCircle(vec2 p, float r) { return length(p) - r; }
float _sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.2 * u_speed;
    p *= u_zoom;
    vec2 c = vec2(cos(t), sin(t * 0.7)) * u_orbit;
    float d = min(_sdCircle(p - c, u_rad),
                  _sdBox(rot(t * u_spin) * p, vec2(u_box)));
    float aa = fwidth(d) + 0.002;
    float mask = 1.0 - smoothstep(0.0, aa, d);
    float rim = smoothstep(0.0, aa, abs(d) - u_rim);
    vec3 col = mix(u_bg, u_fill, mask);
    col = mix(col, u_rimc, rim * (1.0 - mask));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "scene spin speed"},
    "zoom":    {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0,
                "description": "camera zoom"},
    "orbit":   {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.45,
                "description": "circle orbit radius"},
    "rad":     {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.3,
                "description": "circle radius"},
    "box":     {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.28,
                "description": "box half-size"},
    "spin":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": 0.6,
                "description": "box spin rate"},
    "rim":     {"glsl": "float", "min": 0.0, "max": 0.1, "default": 0.04,
                "description": "rim width"},
    "bg":      {"glsl": "color", "default": "#0b0b14", "description": "background"},
    "fill":    {"glsl": "color", "default": "#ff5d73", "description": "shape fill"},
    "rimc":    {"glsl": "color", "default": "#ffe66d", "description": "rim color"},
})