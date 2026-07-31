"""hue_shift_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("hue_shift_gpu", "Hue rotate + saturation of the input (typed)",
          "filter", '''
vec3 _rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 _hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec3 hsv = _rgb2hsv(src);
    hsv.x = fract(hsv.x + u_hue);
    hsv.y = clamp(hsv.y * u_saturation, 0.0, 1.0);
    hsv.z = clamp(hsv.z * u_value, 0.0, 1.0);
    f_color = vec4(_hsv2rgb(hsv), 1.0);
}
''', uniforms={
    "hue":        {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "hue rotation (0..1)"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                   "description": "saturation gain"},
    "value":      {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                   "description": "brightness gain"},
})