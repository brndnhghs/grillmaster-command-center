"""r2_dither_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── P0.4 typed-uniform filter twins (gap nodes 529 / 923 / 462) ──
# Live-preview GPU twins for per-pixel CPU filter nodes whose core math is a
# faithful closed-form f(uv). CPU node stays authoritative for export (two-tier
# precision); these bodies only drive the browser live preview. Each maps the
# node's REAL numeric params onto named u_<var> uniforms (contract #5/#6) via
# CLIENT_GPU_SHIMS. Choice/string params (mode/palette/source/matcap/scene/
# bg_mode/anim_mode) are intentionally left unmapped — the twin hardcodes a
# sensible default and the CPU export honours the exact choice. `step` is the
# prologue-reserved vec2 (pitfall #15b); neighbour offsets use `step.x/step.y`.
# These are filter twins (wired IMAGE in -> shaded IMAGE out), so they render
# BLACK with no input_image (pitfall #10c) — verified with a synthetic input.

# ── 529 R2 Dither (low-discrepancy ordered dither) ──
_register("r2_dither_gpu", "R2 Dither (client-GPU twin of node 529)", "filter", _filter_typed('''
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    lum = clamp(0.5 + (lum - 0.5) * u_contrast, 0.0, 1.0);
    lum = clamp(pow(lum, 1.0 / u_gamma), 0.0, 1.0);
    // R2 low-discrepancy threshold map (Martin Roberts 2018), screen-space
    float gx = (uv.x * u_resolution.x + 0.5) * 0.6180339887;
    float gy = (uv.y * u_resolution.y + 0.5) * 0.5537133391;
    float thr = fract(gx + gy);
    float levels = max(2.0, floor(u_levels + 0.5));
    vec3 outc;
    if (levels <= 2.0) {
        float v = lum > thr ? 1.0 : 0.0;
        outc = vec3(v);
    } else {
        float stepf = 1.0 / (levels - 1.0);
        float bucket = floor(lum / stepf);
        float frac = (lum - bucket * stepf) / stepf;
        float v = (bucket + (frac > thr ? 1.0 : 0.0)) * stepf;
        outc = vec3(clamp(v, 0.0, 1.0));
    }
    f_color = vec4(outc, 1.0);
'''), uniforms={
    "levels": {"glsl": "float", "min": 2.0, "max": 8.0, "default": 2.0, "description": "output quantization levels (2=binary)"},
    "contrast": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0, "description": "source contrast boost"},
    "gamma": {"glsl": "float", "min": 0.3, "max": 2.5, "default": 1.0, "description": "source gamma"},
})