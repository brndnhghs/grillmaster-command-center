"""sobel_dir_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _DERIV_GPU



_register("sobel_dir_typed", "Sobel gradient direction (HSL flow) of the input (typed, node 259)",
          "filter", _DERIV_GPU + '''vec3 _hue2rgb(float h) {
    float k = mod(h * 6.0, 6.0);
    float x = clamp(abs(mod(k, 2.0) - 1.0), 0.0, 1.0);
    if (k < 1.0) return vec3(1.0, x, 0.0);
    if (k < 2.0) return vec3(x, 1.0, 0.0);
    if (k < 3.0) return vec3(0.0, 1.0, x);
    if (k < 4.0) return vec3(0.0, x, 1.0);
    if (k < 5.0) return vec3(x, 0.0, 1.0);
    return vec3(1.0, 0.0, x);
}
void main() {
    vec2 px = u_texel / u_resolution;
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t + tr - bl - 2.0*b - br;
    float ang = atan(gy, gx);                 // [-pi, pi]
    float hue = (ang + 3.14159265) / 6.2831853;
    float mag = clamp(length(vec2(gx, gy)) * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(vec3(u_flat), _hue2rgb(hue), mag);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
              "description": "direction gain"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "flat":  {"glsl": "color", "default": "#101018", "description": "flat-region color"},
})