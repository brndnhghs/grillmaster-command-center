"""threshold_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("threshold_gpu", "Luminance threshold / two-tone of the input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    float l = dot(src, vec3(0.299, 0.587, 0.114));
    float e = smoothstep(u_threshold - u_softness, u_threshold + u_softness, l);
    vec3 col = mix(u_low, u_high, e);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "threshold": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "luminance cutoff"},
    "softness":  {"glsl": "float", "min": 0.0, "max": 0.5, "default": 0.05,
                  "description": "edge softness"},
    "low":       {"glsl": "color", "default": "#0a0a12", "description": "below-threshold color"},
    "high":      {"glsl": "color", "default": "#ffffff", "description": "above-threshold color"},
})