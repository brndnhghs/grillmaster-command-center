"""nebula_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("nebula_typed", "Procedural nebula with typed scale/warp/colors (node 255)",
          "procedural", '''void main() {
    vec2 uv = v_uv * max(u_scale, 0.5);
    float t = u_time * 0.03 * max(u_warp, 0.1);
    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));
    vec2 r = vec2(fbm(uv + 3.0 * q + vec2(1.7, 9.2) + t * 0.3),
                  fbm(uv + 3.0 * q + vec2(8.3, 2.8) + t * 0.4));
    float v = fbm(uv + 3.0 * r);
    float mask = 1.0 - abs(v_uv.y - 0.5) * 2.0 * u_vignette;
    vec3 col = u_shadow + (u_highlight - u_shadow) * (0.5 + 0.5 * cos(v * 4.0 + vec3(0, 1, 2)));
    col *= mask;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "scale":     {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.0,
                  "description": "turbulence scale"},
    "warp":      {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                  "description": "domain-warp / drift speed"},
    "vignette":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                  "description": "vertical vignette"},
    "shadow":    {"glsl": "color", "default": "#0a0a1f", "description": "dark cloud"},
    "highlight": {"glsl": "color", "default": "#5aa0ff", "description": "nebula glow"},
})