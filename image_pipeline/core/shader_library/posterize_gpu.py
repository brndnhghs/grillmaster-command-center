"""posterize_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("posterize_gpu", "Posterize / reduce color levels of input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    float levels = max(float(u_levels), 2.0);
    vec3 q = floor(src * levels + 0.5) / levels;
    if (u_gamma != 1.0) q = pow(clamp(q, 0.0, 1.0), vec3(u_gamma));
    q = mix(src, q, clamp(u_amount, 0.0, 1.0));
    f_color = vec4(q, 1.0);
}
''', uniforms={
    "levels": {"glsl": "int", "min": 2, "max": 32, "default": 5,
               "description": "color levels per channel"},
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
               "description": "effect amount"},
    "gamma":  {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0,
               "description": "post-posterize gamma"},
})