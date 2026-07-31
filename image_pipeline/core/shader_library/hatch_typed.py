"""hatch_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("hatch_typed", "Cross-hatch engraving shading over procedural luminance (node 293)",
          "procedural", '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float lum = fbm(p * u_freq + u_time * 0.04 * u_speed);
    float ang = radians(u_angle);
    vec2 dir = vec2(cos(ang), sin(ang));
    float h1 = step(0.5, fract(dot(p, dir) * u_density));
    vec2 dir2 = vec2(cos(ang + 1.5707963), sin(ang + 1.5707963));
    float h2 = step(0.5, fract(dot(p, dir2) * u_density));
    float ink = (1.0 - lum) * h1;
    ink = max(ink, (1.0 - lum * 0.5) * h2 * step(0.5, lum));
    vec3 col = mix(u_paper, u_ink, clamp(ink, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "luminance drift"},
    "freq":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
                "description": "shading feature size"},
    "angle":   {"glsl": "float", "min": 0.0, "max": 90.0, "default": 35.0,
                "description": "hatch angle (deg)"},
    "density": {"glsl": "float", "min": 4.0, "max": 80.0, "default": 28.0,
                "description": "line density"},
    "paper":   {"glsl": "color", "default": "#f2efe2", "description": "paper"},
    "ink":     {"glsl": "color", "default": "#15110c", "description": "ink"},
})