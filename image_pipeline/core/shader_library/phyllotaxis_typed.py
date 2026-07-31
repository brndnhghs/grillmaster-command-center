"""phyllotaxis_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# ── Categorical coverage pt.8 (typed closed-form patterns, nodes 277-282) ──
# phyllotaxis dots, guilloché engraving, Lissajous trace, radial wave
# interference, curl-noise flow field, kaleidoscopic petal bloom. Each is a
# pure f(uv, t) → exact CPU/GPU parity (P0.6). Continuous-time motion only.

_register("phyllotaxis_typed", "Phyllotaxis: golden-angle sunflower dot spiral (typed, node 277)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    float ga = 2.39996323;
    float best = 1e9;
    float bestk = 0.0;
    int N = int(u_count);
    for (int i = 0; i < 512; i++) {
        if (i >= N) break;
        float fi = float(i);
        float r = u_spread * sqrt(fi) / sqrt(float(N));
        float ang = fi * ga + t;
        vec2 pc = vec2(cos(ang), sin(ang)) * r;
        float d = length(uv - pc);
        if (d < best) { best = d; bestk = fi / float(N); }
    }
    float dot = smoothstep(u_dotsize, u_dotsize * 0.4, best);
    vec3 col = mix(u_bg, inferno(bestk), dot);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "count":   {"glsl": "int", "min": 16, "max": 512, "default": 240,
                "description": "seed count"},
    "spread":  {"glsl": "float", "min": 0.2, "max": 1.2, "default": 0.85,
                "description": "spiral radius"},
    "dotsize": {"glsl": "float", "min": 0.004, "max": 0.06, "default": 0.02,
                "description": "dot radius"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "rotation speed"},
    "bg":      {"glsl": "color", "default": "#05070e", "description": "background"},
})