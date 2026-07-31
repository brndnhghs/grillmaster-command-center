"""box_blur_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed filter / color-grade nodes (ids 244-249) ──────────────────────
# Categorical coverage pt.4 (2026-07-11): the per-pixel filter / color-grade
# family with NAMED typed controls + wireable SCALAR ports — box blur, unsharp
# sharpen, vignette, luminance threshold, hue rotate, and ordered dither.
# Filters take image_in: IMAGE. CPU fns stay authoritative; additive layer.

_register("box_blur_gpu", "Box blur of the input (typed radius/samples)",
          "filter", '''
void main() {
    vec2 px = u_radius / u_resolution;
    int n = int(clamp(float(u_samples), 1.0, 6.0));
    vec3 acc = vec3(0.0); float wsum = 0.0;
    for (int j = -6; j <= 6; j++) {
        if (j < -n || j > n) continue;
        for (int i = -6; i <= 6; i++) {
            if (i < -n || i > n) continue;
            acc += texture(u_texture, v_uv + px * vec2(float(i), float(j))).rgb;
            wsum += 1.0;
        }
    }
    vec3 blurred = acc / max(wsum, 1.0);
    vec3 src = texture(u_texture, v_uv).rgb;
    f_color = vec4(mix(src, blurred, clamp(u_amount, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "radius":  {"glsl": "float", "min": 0.5, "max": 12.0, "default": 2.0,
                "description": "sample spacing (px)"},
    "samples": {"glsl": "int", "min": 1, "max": 6, "default": 3,
                "description": "kernel half-width"},
    "amount":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                "description": "blend amount"},
})