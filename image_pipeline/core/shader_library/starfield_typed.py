"""starfield_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("starfield_typed", "Parallax starfield with twinkling (typed, node 269)",
          "procedural", '''float _sfield(vec2 uv, float seed) {
    vec2 g = floor(uv);
    vec2 f = fract(uv);
    float h = hash21(g + seed);
    float star = smoothstep(0.5 - u_star_size, 0.5 - u_star_size * 0.4,
                            distance(f, vec2(h, fract(h * 13.3))));
    return star;
}
void main() {
    vec2 uv = v_uv * u_density;
    float t = u_time * 0.1 * u_twinkle;
    float total = 0.0;
    vec3 col = u_bg_color;
    for (int i = 1; i <= 4; i++) {
        float fi = float(i);
        float depth = fi / 4.0;
        vec2 suv = (uv * depth) + vec2(t * depth, t * depth * 0.3) + fi * 17.0;
        float s = _sfield(suv, fi * 3.7);
        float tw = 0.6 + 0.4 * sin(t * (2.0 + fi) + hash21(suv) * 6.2831);
        total += s * tw * (1.0 - depth * 0.4);
        col += u_star_color * s * tw * (1.0 - depth * 0.5);
    }
    col = max(col, u_bg_color);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "density":     {"glsl": "float", "min": 4.0, "max": 80.0, "default": 30.0,
                    "description": "stars per screen"},
    "star_size":   {"glsl": "float", "min": 0.01, "max": 0.2, "default": 0.06,
                    "description": "star radius"},
    "twinkle":     {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                    "description": "twinkle animation speed"},
    "bg_color":    {"glsl": "color", "default": "#03040a", "description": "sky color"},
    "star_color":  {"glsl": "color", "default": "#ffffff", "description": "star color"},
})