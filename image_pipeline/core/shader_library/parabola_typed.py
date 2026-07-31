"""parabola_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# 306 — Confocal parabola family (op-art): sum of parabolas f(s^2/c) over N
# directions, coloured with inferno. Distinct from the domain-warp grid.
_register("parabola_typed", "Confocal parabola family — op-art interference (typed, node 306)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time * u_speed;
    int N = int(max(u_arms, 1.0));
    float acc = 0.0;
    for (int k = 0; k < 48; k++) {
        if (k >= N) break;
        float a = (float(k) / float(N)) * 3.14159265;
        vec2 dir = vec2(cos(a), sin(a));
        float s = dot(p, dir);
        float c = dot(p, vec2(-dir.y, dir.x));
        acc += cos((s * s / max(abs(c), 0.05)) * u_freq + t);
    }
    acc /= float(N);
    vec3 col = inferno(0.5 + 0.5 * acc);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "arms":  {"glsl": "float", "min": 2.0, "max": 24.0, "default": 8.0, "description": "parabola directions"},
    "freq":  {"glsl": "float", "min": 1.0, "max": 30.0, "default": 8.0, "description": "curvature frequency"},
    "scale": {"glsl": "float", "min": 2.0, "max": 12.0, "default": 6.0, "description": "spatial scale"},
    "speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "animation speed"},
})