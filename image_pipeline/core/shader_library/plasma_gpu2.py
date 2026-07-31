"""plasma_gpu2 — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed-uniform nodes 226-231 (categorical coverage expansion) ──────
# Each declares its variables via `uniforms=` so the node factory exposes them
# as real params (sliders / color pickers / dropdowns) AND wireable SCALAR
# input ports, with data-typed outputs (image: IMAGE, luminance: FIELD). Bodies
# stay in the GL330/ES300-compatible parity subset (prologue helpers + no
# forbidden tokens).

_register("plasma_gpu2", "Animated plasma with typed scale/colors/warp",
          "procedural", '''
void main() {
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * u_scale;
    // Slow, smooth multi-octave drift (no discrete cusps).
    float t = u_time * u_speed * 0.25;
    float v = sin(p.x * 6.0 + t) * cos(p.y * 4.0 + t * 0.7);
    v += sin(p.x * 11.0 - t * 1.2) * cos(p.y * 9.0 + t * 0.5) * 0.6;
    v += sin((p.x + p.y) * 16.0 + t * 0.3) * 0.3;
    if (u_warp > 0.001) {
        v += 0.4 * sin(length(p) * u_warp * 8.0 - t * 1.5);
    }
    v = v * 0.5 + 0.5;
    v = clamp(pow(v, max(u_contrast, 0.05)), 0.0, 1.0);
    f_color = vec4(mix(u_color_a, u_color_b, v), 1.0);
}
''', uniforms={
    "scale":    {"glsl": "float", "min": 0.5, "max": 16.0, "default": 4.0,
                 "description": "plasma spatial scale"},
    "speed":    {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                 "description": "animation speed"},
    "warp":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                 "description": "radial warp amount"},
    "contrast": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                 "description": "output contrast (gamma)"},
    "color_a":  {"glsl": "color", "default": "#10071f", "description": "low color"},
    "color_b":  {"glsl": "color", "default": "#ffcf4d", "description": "high color"},
})