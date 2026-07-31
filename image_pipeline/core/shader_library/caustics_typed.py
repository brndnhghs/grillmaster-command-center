"""caustics_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("caustics_typed", "Animated water caustics (typed, node 296)",
          "procedural", '''vec3 _hsv(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
void main() {
    vec2 p = v_uv * u_scale;
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    vec2 w = vec2(0.0);
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        w += vec2(sin(p.y * (1.0 + fi * 0.3) + t + fi),
                  cos(p.x * (1.0 + fi * 0.3) - t * 1.1 + fi * 1.7));
        p *= 1.4;
    }
    float c = 1.0 - abs(sin(w.x + w.y) * 0.5 + 0.5);
    c = pow(c, u_sharp);
    vec3 col = mix(u_deep, u_shallow, c);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "ripple speed"},
    "scale":    {"glsl": "float", "min": 1.0, "max": 16.0, "default": 6.0,
                "description": "ripple density"},
    "sharp":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 4.0,
                "description": "caustic sharpness"},
    "deep":     {"glsl": "color", "default": "#02121f", "description": "deep water"},
    "shallow":  {"glsl": "color", "default": "#4fd6ff", "description": "lit water"},
})