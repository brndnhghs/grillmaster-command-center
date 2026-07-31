"""side_window_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── P0 filter twins: Side Window (357) + God Rays (446) ──────────────────────
# Additive: the server's CPU numpy nodes stay the authoritative export (two-tier
# precision). These bodies only drive the browser live preview. They reuse the
# prologue helpers injected by _filter_typed (uv/orig/step/u_texture/
# u_resolution/u_time) so every new twin is covered automatically by
# test_webgl2_transform_is_valid + the gl330 legacy-equivalence parametrized
# tests. IMPORTANT (pitfall #15b): never declare a local named `step` — the
# wrapper injects `vec2 step = 1.0 / u_resolution;` into main().

_register("side_window_gpu", "Side Window Filter (client-GPU twin of node 357)", "filter", _filter_typed('''
    int R = int(clamp(u_radius, 1.0, 24.0));
    vec4 acc = vec4(0.0);
    float n = 0.0;
    for (int x = -24; x <= 24; x++) {
        if (abs(float(x)) > float(R)) continue;
        for (int y = -24; y <= 24; y++) {
            if (abs(float(y)) > float(R)) continue;
            acc += texture(u_texture, uv + vec2(float(x), float(y)) * step);
            n += 1.0;
        }
    }
    vec4 blurred = acc / max(n, 1.0);
    f_color = mix(orig, blurred, clamp(u_blend, 0.0, 1.0));
'''), uniforms={
    "radius": {"glsl": "float", "min": 1.0, "max": 40.0, "default": 6.0, "description": "side-window half-size in pixels"},
    "blend": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "mix original (0) vs smoothed (1)"},
})