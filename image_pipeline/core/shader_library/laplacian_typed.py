"""laplacian_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _DERIV_GPU



_register("laplacian_typed", "Laplacian zero-crossing / edge field of the input (typed, node 260)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float c  = _dlum(v_uv);
    float l  = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float r  = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float t  = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float b  = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float lap = (l + r + t + b - 4.0 * c);
    float e = clamp(abs(lap) * u_gain * 0.5, 0.0, 1.0);
    vec3 col = mix(u_flat, u_edge, e);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "gain":  {"glsl": "float", "min": 0.2, "max": 6.0, "default": 2.0,
              "description": "laplacian gain"},
    "texel": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
              "description": "kernel spacing (px)"},
    "flat":  {"glsl": "color", "default": "#080810", "description": "flat color"},
    "edge":  {"glsl": "color", "default": "#ff5cf0", "description": "edge color"},
})