"""gradient_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── P0.5 LUT / color client-GPU twins (client-GPU live preview of nodes
# 10/11/39/77) ───────────────────────────────────────────────────────────────
# Additive: the server's CPU numpy nodes stay the authoritative export (two-tier
# precision). These bodies only drive the browser live preview. They reuse the
# prologue helpers (rot/hash21/noise/fbm/_INFERNO) so every new twin is covered
# automatically by test_webgl2_transform_is_valid + the gl330 legacy-equivalence
# parametrized tests.
#
# IMPORTANT (pitfall #15b): filter twins must NOT declare a local named `step` —
# the _filter_shader wrapper injects `vec2 step = 1.0 / u_resolution;` into
# main(). Use `px` / `gstep` / `cell_sz` instead.

_register("gradient_gpu",
          "Gradient generator (client-GPU twin of node 11)",
          "procedural", '''

// sRGB-ish gradient between two endpoint colors expressed as HSV-ish offsets.
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // u_params.x = direction (radians, 0.5 = 0 rad; maps -PI..PI),
    // u_params.y = center_x (0.5 = middle), u_params.z = center_y,
    // u_params.w = gradient_type (0=linear,1=radial,2=concentric,3=angular,4=diamond).
    float dir = (u_cx - 0.5) * 6.2831853;
    vec2 ctr = vec2(u_cy, 0.5);
    vec2 p = v_uv - ctr;

    float t;
    int gtype = int(floor(0.5 * 4.999));
    if (gtype == 1) {                       // radial
        t = length(p);
    } else if (gtype == 2) {                // concentric (ring index)
        t = fract(length(p) * 8.0 + u_time * 0.05);
    } else if (gtype == 3) {                // angular
        float a = atan(p.y, p.x) - dir;
        t = 0.5 + 0.5 * (a / 3.14159265);
    } else if (gtype == 4) {                // diamond
        t = abs(p.x) + abs(p.y);
    } else {                                // linear
        t = 0.5 + 0.5 * dot(normalize(vec2(cos(dir), sin(dir)) + 1e-5), p);
    }
    t = clamp(t, 0.0, 1.0);

    // Two endpoint hues (cyan -> orange, echoing the node's color1/color2 defaults).
    vec3 c1 = hsv2rgb(vec3(0.62, 0.80, 0.55));
    vec3 c2 = hsv2rgb(vec3(0.05, 0.85, 0.95));
    vec3 col = mix(c1, c2, t);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "cx": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "center x"},
    "cy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "center y"}
}
    )