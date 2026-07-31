"""vignette_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("vignette_gpu", "Vignette darkening of the input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec2 d = v_uv - 0.5;
    d.x *= u_resolution.x / u_resolution.y;
    float r = length(d) * 1.41421356;
    float v = smoothstep(u_outer, u_inner, r);
    v = mix(1.0, v, clamp(u_amount, 0.0, 1.0));
    f_color = vec4(mix(u_color, src, v), 1.0);
}
''', uniforms={
    "inner":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.3,
               "description": "inner (bright) radius"},
    "outer":  {"glsl": "float", "min": 0.2, "max": 1.6, "default": 1.0,
               "description": "outer (dark) radius"},
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8,
               "description": "vignette amount"},
    "color":  {"glsl": "color", "default": "#000000", "description": "vignette color"},
})