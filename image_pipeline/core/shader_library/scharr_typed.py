"""scharr_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _DERIV_GPU



_register("scharr_typed", "Scharr operator (sharper Sobel) magnitude (typed, node 261)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float tl = _dlum(v_uv + px * vec2(-1.0,  1.0));
    float  l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float bl = _dlum(v_uv + px * vec2(-1.0, -1.0));
    float  t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float  b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float tr = _dlum(v_uv + px * vec2( 1.0,  1.0));
    float  r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float br = _dlum(v_uv + px * vec2( 1.0, -1.0));
    float gx = -3.0*tl - 10.0*l - 3.0*bl + 3.0*tr + 10.0*r + 3.0*br;
    float gy =  3.0*tl + 10.0*t + 3.0*tr - 3.0*bl - 10.0*b - 3.0*br;
    float m = clamp(length(vec2(gx, gy)) * u_gain * 0.0625, 0.0, 1.0);
    vec3 col = mix(u_low, u_high, m);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.6,
              "description": "magnitude gain"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "low":   {"glsl": "color", "default": "#001018", "description": "low color"},
    "high":  {"glsl": "color", "default": "#4dffd0", "description": "edge color"},
})