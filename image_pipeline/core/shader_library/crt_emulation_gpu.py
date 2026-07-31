"""crt_emulation_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed




# ── 522 CRT Emulation (client-GPU twin) ──
_register("crt_emulation_gpu",
          "CRT Emulation (client-GPU twin of node 522)",
          "filter", _filter_typed('''
    // Barrel distortion (curvature), aperture-grille mask, scanlines, edge
    // vignette, RGB chroma shift, and a u_time-driven vertical roll + brightness
    // flicker so the live preview is animated (cos term, not sin, to avoid the
    // 0/pi phase degeneracy).
    vec2 p = uv * 2.0 - 1.0;
    float r2 = dot(p, p);
    vec2 quv = (p * (1.0 + u_curvature * r2)) * 0.5 + 0.5;
    quv.y = fract(quv.y + u_time * 0.05 * u_roll_speed);
    vec3 col;
    col.r = sample(clamp(quv + vec2(u_chroma * 0.01 * (quv.x - 0.5), 0.0), 0.0, 1.0)).r;
    col.g = sample(clamp(quv, 0.0, 1.0)).g;
    col.b = sample(clamp(quv - vec2(u_chroma * 0.01 * (quv.x - 0.5), 0.0), 0.0, 1.0)).b;
    float scan = 0.5 + 0.5 * sin(quv.y * u_resolution.y * 0.5 * u_scan_freq);
    col *= 1.0 - u_scanline * (1.0 - scan);
    float m = 0.5 + 0.5 * cos(quv.x * u_resolution.x * 1.04719755);
    col *= 1.0 - u_mask_strength * (1.0 - m);
    col *= 1.0 - u_vignette * r2;
    col *= u_brightness * (1.0 - u_flicker * (0.5 + 0.5 * cos(u_time * 7.0)));
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
'''), uniforms={
    "curvature":  {"glsl": "float", "min": 0.0, "max": 0.45, "default": 0.18, "description": "barrel distortion amount"},
    "scanline":   {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.35, "description": "scanline darkness"},
    "scan_freq":  {"glsl": "float", "min": 1.0, "max": 8.0,  "default": 2.5,  "description": "scanline frequency"},
    "mask_strength": {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.35, "description": "aperture-grille mask strength"},
    "vignette":   {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.3,  "description": "edge vignette"},
    "chroma":     {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.25, "description": "RGB chroma shift"},
    "roll_speed": {"glsl": "float", "min": 0.0, "max": 3.0,  "default": 1.0,  "description": "vertical roll speed"},
    "flicker":    {"glsl": "float", "min": 0.0, "max": 0.3,  "default": 0.06, "description": "brightness flicker"},
    "brightness": {"glsl": "float", "min": 0.4, "max": 2.0,  "default": 1.1,  "description": "overall brightness"},
})