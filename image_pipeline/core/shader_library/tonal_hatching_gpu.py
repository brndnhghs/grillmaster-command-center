"""tonal_hatching_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# 339 Tonal Hatching — pen-and-ink crosshatch screening (GPU live twin of
# node 339, Winkenbach & Salesin 1994). Mirrors the CPU node's geometry:
# ink coverage f = 1 - L^contrast, then screen `layers` rotated stripe
# masks (each layer rotated 180/layers deg) turning on where f exceeds a
# per-layer threshold; darkest tones collapse to solid ink. The CPU fn
# stays authoritative for exact W&D paper/ink palettes + animation modes;
# the twin covers the dominant light-paper / black-ink look live. Numeric
# node params (spacing/line_width/layers/angle/contrast) are mapped by
# name (contract #5); string choices `paper`/`ink_tone` are left
# unmapped (pitfall #14) so the preview uses the canonical light/black.
# NOTE: do NOT redeclare `step` (the wrapper provides a vec2 `step`) — use
# plain comparisons for the stripe test.
_register("tonal_hatching_gpu", "GPU tonal hatching (crosshatch pen-and-ink screening)", "filter", _filter_typed('''
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    lum = clamp(lum, 0.0, 1.0);
    float contrast = max(0.1, u_contrast);
    lum = pow(lum, contrast);
    float f = 1.0 - lum;                          // ink coverage fraction
    float spacing = max(3.0, u_spacing);          // px between strokes
    float lw = clamp(u_line_width, 0.5, spacing - 1.0);
    float nlayers = max(1.0, min(4.0, floor(u_layers + 0.5)));
    float baseA = radians(u_angle);
    vec2 px = uv * u_resolution;
    bool ink = false;
    for (int i = 0; i < 4; i++) {
        if (float(i) >= nlayers) break;
        float thr = (float(i) + 1.0) / (nlayers + 1.0);
        float ang = baseA + float(i) * (3.14159265 / nlayers);
        float ca = cos(ang), sa = sin(ang);
        float lyr = px.x * sa + px.y * ca;
        float phase = mod(lyr, spacing);
        if (f > thr && phase < lw) ink = true;
    }
    if (f > 0.9) ink = true;                      // darkest tones solid ink
    vec3 paper_col = vec3(0.97, 0.96, 0.92);
    vec3 ink_col = vec3(0.09, 0.09, 0.11);
    f_color = vec4(mix(paper_col, ink_col, ink ? 1.0 : 0.0), 1.0);
'''), uniforms={
    "spacing": {"glsl": "float", "min": 4.0, "max": 24.0, "default": 10.0, "description": "hatch line spacing (px)"},
    "line_width": {"glsl": "float", "min": 0.5, "max": 5.0, "default": 1.5, "description": "hatch line width (px)"},
    "layers": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 3.0, "description": "crosshatch layer count"},
    "angle": {"glsl": "float", "min": 0.0, "max": 180.0, "default": 45.0, "description": "base hatch angle (deg)"},
    "contrast": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "luminance gamma"},
})