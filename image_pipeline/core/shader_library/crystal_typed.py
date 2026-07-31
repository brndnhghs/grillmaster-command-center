"""crystal_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# 304 — Crystal diffraction: sum of N cosinusoidal gratings evenly fanned around
# the circle, coloured with the inferno map. Rotating the fan gives the
# classic X-ray-diffraction look.
_register("crystal_typed", "Crystal diffraction — sum of N sinusoidal gratings (typed, node 304)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time * u_speed;
    int N = int(max(u_arms, 1.0));
    float acc = 0.0;
    for (int k = 0; k < 64; k++) {
        if (k >= N) break;
        float a = (float(k) / float(N)) * 6.2831853 + u_rotation + t * 0.15;
        vec2 dir = vec2(cos(a), sin(a));
        acc += cos(dot(p, dir) * u_freq);
    }
    acc /= float(N);
    float v = 0.5 + 0.5 * acc;
    vec3 col = inferno(clamp(v, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "arms":     {"glsl": "float", "min": 2.0, "max": 24.0, "default": 6.0, "description": "grating directions"},
    "freq":     {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0, "description": "grating frequency"},
    "scale":    {"glsl": "float", "min": 2.0, "max": 12.0, "default": 6.0, "description": "spatial scale"},
    "rotation": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0, "description": "base rotation"},
    "speed":    {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "animation speed"},
})