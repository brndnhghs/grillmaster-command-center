"""interference_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("interference_typed", "Radial wave interference from N sources (typed, node 280)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.5 * u_speed;
    float acc = 0.0;
    int N = int(u_sources);
    for (int i = 0; i < 8; i++) {
        if (i >= N) break;
        float ang = float(i) / float(N) * 6.28318530;
        vec2 src = vec2(cos(ang), sin(ang)) * u_radius;
        float d = length(p - src);
        acc += sin(d * u_freq * 6.28318530 - t);
    }
    float v = 0.5 + 0.5 * acc / float(N);
    v = clamp(pow(v, u_contrast), 0.0, 1.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "sources":  {"glsl": "int", "min": 2, "max": 8, "default": 4,
                 "description": "wave source count"},
    "radius":   {"glsl": "float", "min": 0.1, "max": 0.6, "default": 0.35,
                 "description": "source ring radius"},
    "freq":     {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0,
                 "description": "wave frequency"},
    "contrast": {"glsl": "float", "min": 0.3, "max": 4.0, "default": 1.4,
                 "description": "contrast gamma"},
    "speed":    {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                 "description": "wave speed"},
})