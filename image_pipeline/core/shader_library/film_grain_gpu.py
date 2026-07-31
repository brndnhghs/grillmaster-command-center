"""film_grain_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 489 Film Grain (client-GPU twin) ──
# Luminance-adaptive emulsion grain — closed-form preview of node 489. Per-pixel
# hash grain, shadow-weighted by luminance (real film grain reads stronger in
# shadows). CPU node is authoritative for export; this is the live-preview path.
# `color`/`source`/`palette`/`anim_mode` are CPU-only choices (pitfall #14) — the
# preview animates the grain field with u_time (flicker) so it stays live and
# is_time_varying is honest (no sin-phase degeneracy). intensity/adapt/grain_size
# are wired by name to the shader's u_<name> uniforms (typed-uniform contract).
_register("film_grain_gpu",
          "Film Grain (client-GPU twin of node 489)",
          "filter", _filter_typed('''
    vec3 srgb = orig.rgb;
    // blocky grain: quantize by grain_size, hash per block (+u_time = flicker)
    vec2 cell = floor(uv * u_resolution / max(1.0, u_grain_size));
    float r1 = hash21(cell + vec2(0.123, 0.0) + u_time * 13.0);
    float r2 = hash21(cell + vec2(7.77, 3.33) + u_time * 17.0);
    float r3 = hash21(cell + vec2(2.22, 9.99) + u_time * 11.0);
    vec3 grain = vec3(r1, r2, r3) * 2.0 - 1.0;          // [-1,1]
    float lum = dot(srgb, vec3(0.299, 0.587, 0.114));
    float k = u_intensity * (1.0 + 2.0 * u_adapt * (1.0 - lum));
    vec3 outc = clamp(srgb + grain * k, 0.0, 1.0);
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
'''), uniforms={
    "intensity":  {"glsl": "float", "min": 0.0, "max": 0.6, "default": 0.12, "description": "grain strength / ISO-like amount"},
    "adapt":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.7, "description": "shadow-weighting of grain"},
    "grain_size": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 1.0, "description": "grain pixel scale (chunkier if >1)"},
})