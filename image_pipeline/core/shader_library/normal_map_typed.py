"""normal_map_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _DERIV_GPU



_register("normal_map_typed", "Normal map (bump) from luminance gradient (typed, node 262)",
          "filter", _DERIV_GPU + '''void main() {
    vec2 px = u_texel / u_resolution;
    float l = _dlum(v_uv + px * vec2(-1.0,  0.0));
    float r = _dlum(v_uv + px * vec2( 1.0,  0.0));
    float t = _dlum(v_uv + px * vec2( 0.0,  1.0));
    float b = _dlum(v_uv + px * vec2( 0.0, -1.0));
    float dx = (r - l) * u_strength;
    float dy = (t - b) * u_strength;
    vec3 n = normalize(vec3(-dx, -dy, 1.0));
    vec3 col = n * 0.5 + 0.5;
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": 0.1, "max": 8.0, "default": 2.0,
                 "description": "surface bumpiness"},
    "texel":    {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.5,
                 "description": "sample spacing (px)"},
})