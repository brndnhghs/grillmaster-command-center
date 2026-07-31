"""strange_attractor_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("strange_attractor_typed", "Strange-attractor bands: Clifford map density (typed, node 276)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    vec2 p = uv * 2.6;
    float t = u_time * 0.05 * u_speed;
    float a = u_a + 0.12 * sin(t);
    float b = u_b + 0.12 * cos(t);
    float c = u_c;
    float d = u_d;
    vec2 q = p;
    float acc = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 nx = vec2(sin(a * q.y) + c * cos(a * q.x),
                       sin(b * q.x) + d * cos(b * q.y));
        acc += exp(-dot(nx - p, nx - p) * u_band);
        q = nx;
    }
    float v = clamp(acc * u_gain, 0.0, 1.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "a":    {"glsl": "float", "min": -2.5, "max": 2.5, "default": -1.4,
             "description": "Clifford a"},
    "b":    {"glsl": "float", "min": -2.5, "max": 2.5, "default": 1.6,
             "description": "Clifford b"},
    "c":    {"glsl": "float", "min": -2.0, "max": 2.0, "default": 1.0,
             "description": "Clifford c"},
    "d":    {"glsl": "float", "min": -2.0, "max": 2.0, "default": 0.7,
             "description": "Clifford d"},
    "band": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 18.0,
             "description": "band tightness"},
    "gain": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
             "description": "density gain"},
    "speed":{"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
             "description": "animation speed"},
})