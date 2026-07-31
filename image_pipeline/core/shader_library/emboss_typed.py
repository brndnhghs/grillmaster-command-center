"""emboss_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _DERIV_GPU



_register("emboss_typed", "Directional emboss (relief) of the input (typed, node 264)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    // 3x3 emboss kernel rotated by u_angle
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
    float relief = gx * cos(u_angle) + gy * sin(u_angle);
    float e = clamp(0.5 + relief * u_gain * 0.25, 0.0, 1.0);
    vec3 col = mix(u_dark, u_light, e);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.5,
              "description": "relief strength"},
    "angle": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 2.3561945,
              "description": "light direction (rad)"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel thickness (px)"},
    "dark":  {"glsl": "color", "default": "#161a2a", "description": "shadow color"},
    "light": {"glsl": "color", "default": "#f3f0e6", "description": "highlight color"},
})