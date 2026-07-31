"""gradient_gpu2 — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════
#  TYPED-UNIFORM SHADERS (named vars, no p1..p4)
# ═══════════════════════════════════════════════
#
# Each declares its variables via `uniforms=` — the node factory exposes them
# as real params (sliders / color pickers / dropdowns) AND wireable SCALAR
# ports. Bodies stay in the GL330/ES300-compatible parity subset.

_register("gradient_gpu2", "Gradient with typed controls (linear/radial/conic/diamond)",
          "procedural", '''
void main() {
    vec2 uv = v_uv;
    vec2 ctr = vec2(u_center_x, u_center_y);
    float a = radians(u_angle);
    vec2 dir = vec2(cos(a), sin(a));
    float t;
    if (u_mode == 1) {                       // radial
        t = length(uv - ctr) * 1.41421356;
    } else if (u_mode == 2) {                // conic
        vec2 d = uv - ctr;
        t = fract((atan(d.y, d.x) - a) / 6.28318530 + 1.0);
    } else if (u_mode == 3) {                // diamond
        vec2 d = abs(uv - ctr);
        t = (d.x + d.y) * 1.2;
    } else {                                 // linear
        t = dot(uv - ctr, dir) + 0.5;
    }
    t = clamp(t, 0.0, 1.0);
    if (u_bands > 1.5) t = floor(t * u_bands) / max(u_bands - 1.0, 1.0);  // posterized bands
    // Ordered-dither the ramp to hide 8-bit banding on smooth gradients.
    float dth = (hash21(gl_FragCoord.xy) - 0.5) * u_dither * 0.02;
    t = clamp(t + dth, 0.0, 1.0);
    f_color = vec4(mix(u_color_a, u_color_b, t), 1.0);
}
''', uniforms={
    "mode":     {"glsl": "choice", "choices": ["linear", "radial", "conic", "diamond"],
                 "default": "linear", "description": "gradient geometry"},
    "angle":    {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                 "description": "gradient angle (deg)"},
    "center_x": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                 "description": "center X"},
    "center_y": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                 "description": "center Y"},
    "color_a":  {"glsl": "color", "default": "#0b1026", "description": "start color"},
    "color_b":  {"glsl": "color", "default": "#4a9eff", "description": "end color"},
    "bands":    {"glsl": "float", "min": 0.0, "max": 32.0, "default": 0.0,
                 "description": "posterize bands (0 = smooth)"},
    "dither":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.25,
                 "description": "dither strength (hides banding)"},
})