"""kaleido_bloom_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("kaleido_bloom_typed", "Kaleidoscopic petal bloom (typed, node 282)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.15 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    float seg = 6.28318530 / u_slices;
    a = mod(a + t, seg);
    a = abs(a - seg * 0.5);
    float petal = cos(a * u_slices * 0.5) * sin(r * u_rings * 6.28318530 - t * 2.0);
    float v = clamp(0.5 + 0.5 * petal * u_gain, 0.0, 1.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "slices": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 8.0,
               "description": "mirror slices"},
    "rings":  {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0,
               "description": "radial rings"},
    "gain":   {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.3,
               "description": "intensity gain"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
})