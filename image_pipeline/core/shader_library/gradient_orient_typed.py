"""gradient_orient_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _DERIV_GPU



_register("gradient_orient_typed", "Gradient orientation flow field (typed, node 263)",
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
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy =  tl + 2.0*t + tr - bl - 2.0*b - br;
    vec2 dir = (abs(gx) + abs(gy) < 1e-4) ? vec2(1.0, 0.0) : normalize(vec2(gx, gy));
    // rotate the orientation vector by the wind angle and tint by strength
    float ang = u_wind + u_time * u_spin;
    vec2 d = rot(ang) * dir;
    float mag = clamp(length(vec2(gx, gy)) * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(vec3(u_flat), vec3(d * 0.5 + 0.5, 0.5), mag);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
              "description": "flow strength"},
    "wind":  {"glsl": "float", "min": -3.14159, "max": 3.14159, "default": 0.0,
              "description": "flow rotation (rad)"},
    "spin":  {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0,
              "description": "animated spin speed"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "flat":  {"glsl": "color", "default": "#0a0a12", "description": "flat color"},
})