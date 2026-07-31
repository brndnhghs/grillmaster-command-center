"""vhs_tape_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 527 VHS Tape (client-GPU twin) ──
_register("vhs_tape_gpu",
          "VHS Tape (client-GPU twin of node 527)",
          "filter", _filter_typed('''
    // Chroma smear + per-line chroma shift, horizontal line jitter and a
    // tracking-band distortion driven by u_time, luma noise, and
    // saturation/contrast/brightness grading. cos/linear temporal terms (not
    // sin) keep the live preview honest.
    float n = hash21(vec2(uv.x * 100.0, floor(uv.y * u_resolution.y)));
    float jit = (hash21(vec2(floor(uv.y * 40.0), u_time)) - 0.5) * u_line_jitter * 0.05;
    vec2 quv = vec2(uv.x + jit + (uv.y - 0.5) * u_skew * 0.15,
                    fract(uv.y + u_time * 0.02 * u_roll_speed));
    float off = u_chroma_smear * 0.02 + u_chroma_shift * 0.0005 * sin(uv.y * 80.0 + u_time);
    vec3 col;
    col.r = sample(clamp(quv + vec2(off, 0.0), 0.0, 1.0)).r;
    col.g = sample(clamp(quv, 0.0, 1.0)).g;
    col.b = sample(clamp(quv - vec2(off, 0.0), 0.0, 1.0)).b;
    col += (n - 0.5) * u_luma_noise;
    float track = smoothstep(0.0, 0.08,
        abs(fract(uv.y - u_time * 0.1 * u_roll_speed) - 0.5) - (0.45 - 0.1 * u_tracking));
    col *= 1.0 - track * 0.5;
    float l = dot(col, vec3(0.299, 0.587, 0.114));
    col = (col - l) * u_saturation + l;
    col = (col - 0.5) * u_contrast + 0.5;
    col *= u_brightness;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
'''), uniforms={
    "chroma_smear": {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.55, "description": "horizontal chroma smear"},
    "chroma_shift": {"glsl": "float", "min": 0.0, "max": 24.0, "default": 8.0,  "description": "per-line chroma offset"},
    "luma_noise":   {"glsl": "float", "min": 0.0, "max": 0.6,  "default": 0.12, "description": "luma noise amount"},
    "line_jitter":  {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.45, "description": "horizontal line jitter"},
    "tracking":     {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.5,  "description": "tracking band strength"},
    "roll_speed":     {"glsl": "float", "min": 0.0, "max": 3.0,  "default": 1.0,  "description": "vertical roll speed"},
    "skew":         {"glsl": "float", "min": 0.0, "max": 1.0,  "default": 0.35, "description": "tape skew"},
    "saturation":   {"glsl": "float", "min": 0.0, "max": 2.0,  "default": 1.25, "description": "saturation"},
    "contrast":     {"glsl": "float", "min": 0.3, "max": 2.0,  "default": 1.1,  "description": "contrast"},
    "brightness":   {"glsl": "float", "min": 0.4, "max": 2.0,  "default": 1.05, "description": "brightness"},
})