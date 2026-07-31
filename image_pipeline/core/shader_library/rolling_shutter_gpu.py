"""rolling_shutter_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 430 Rolling Shutter (scan-line shear + jello bow) ──
# Live-preview GPU twin for per-pixel CPU filter node 430 (Rolling Shutter).
# CPU numpy node 430 stays authoritative for export (two-tier precision).
# Faithful closed-form horizontal-scan warp: each row is shifted in X by
# skew*tau and bowed in Y by wobble*sin(2π*freq*tau+phase)*(x/W-0.5) — BOTH
# axes (CPU _warp, lines 189-197). GLSL uv.y=0 is bottom; CPU scans top->bottom
# so tau = 1.0 - uv.y. `direction` (choice) is hardcoded horizontal in the twin;
# `source`/`palette`/`noise_amp`/`blur_sigma` (non-local/global) and
# `anim_mode`/`anim_speed`/`time` are left unmapped. Wobble phase is driven by
# u_time so the live preview animates (liveness); CPU export honours anim_mode.
# Renders BLACK with no input_image (pitfall #10c) — verified with a synthetic
# input. Does NOT redeclare `step` (pitfall #15b).
_register("rolling_shutter_gpu", "Rolling Shutter (client-GPU twin of node 430)", "filter", _filter_typed('''
    float tA = u_time;
    float tau = 1.0 - uv.y;                                  // top row tau=0, bottom tau=1
    float dx = (u_skew + 0.2 * sin(tA)) * tau;               // base shear + time breathing (0 at t=0)
    float wave = sin(6.2831853 * u_wobble_freq * tau + tA);
    float dy = u_wobble * wave * (uv.x - 0.5);               // jello bow in Y
    f_color = texture(u_texture, clamp(uv + vec2(dx, dy), 0.0, 1.0));
'''), uniforms={
    "skew":        {"glsl": "float", "min": -1.0, "max": 1.0,  "default": 0.3,  "description": "scan-line shear amount (fraction of width)"},
    "wobble":      {"glsl": "float", "min": 0.0,  "max": 0.6,  "default": 0.15, "description": "jello bend amplitude"},
    "wobble_freq": {"glsl": "float", "min": 0.5,  "max": 10.0, "default": 3.0,  "description": "jello bend spatial frequency"},
})