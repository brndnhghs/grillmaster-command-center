"""cmyk_halftone_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── CMYK Halftone (node 399) client-GPU twin ──
# Classic print color-separation halftone. The source image is split into
# C/M/Y/K channels; each channel is screened onto its own rotated dot grid and
# the ink layers are recombined subtractively on paper. Per-channel screening is
# a closed-form function of (uv, input, params) -> exact parity preview, no
# seeded-layout divergence (same family as 53/43/57/172). The CPU numpy node
# stays authoritative for export (two-tier precision).
# Real numeric params bound via CLIENT_GPU_SHIMS param_map (spacing/max_dot/
# angle_offset); ink_set/paper/source/anim_mode are choice strings (pitfall
# #14) left unmapped — the preview renders the node's default cmyk ink set on
# white paper. pitfall #15: 0.5 -> node default so a neutral u_params yields the
# canonical view.
_register("cmyk_halftone_gpu",
          "CMYK halftone — classic print color-separation screening (client-GPU twin of node 399)",
          "filter", '''
float halftone_dot(vec2 uv, vec2 res, float value, float angleDeg,
                  float spacing, float maxDot) {
    float ang = radians(angleDeg);
    vec2 p = uv * res;                       // pixel coords
    vec2 g = rot(ang) * p;                   // rotate into screen space
    vec2 cell = mod(g, spacing) - spacing * 0.5;
    float d = length(cell);
    float r = sqrt(clamp(value, 0.0, 1.0)) * maxDot * spacing * 0.5;
    // 1.0 inside the dot, soft 1px edge for AA.
    return 1.0 - smoothstep(r, r + 1.0, d);
}
void main() {
    // Decode real node params via the typed uniforms (u_<name>) — these are
    // driven by the node's spacing/max_dot/angle_offset params on the client
    // AND passed as named_params on the server.
    //   spacing [2,40] ; max_dot [0.3,1.4] ; angle_offset [-45,45] deg
    float spacing  = mix(2.0, 40.0, clamp(u_spacing, 0.0, 1.0));
    float maxDot   = mix(0.3, 1.4, clamp(u_max_dot, 0.0, 1.0));
    float angleOff = (u_angle_offset - 0.5) * 90.0;   // -45..45 deg

    vec3 src = texture(u_texture, v_uv).rgb;
    float luma = dot(src, vec3(0.299, 0.587, 0.114));

    // Standard CMYK screen angles (C 15, M 75, Y 0, K 45) + global offset.
    float c = halftone_dot(v_uv, u_resolution, src.r, 15.0 + angleOff, spacing, maxDot);
    float m = halftone_dot(v_uv, u_resolution, src.g, 75.0 + angleOff, spacing, maxDot);
    float y = halftone_dot(v_uv, u_resolution, src.b,  0.0 + angleOff, spacing, maxDot);
    float k = halftone_dot(v_uv, u_resolution, 1.0 - luma, 45.0 + angleOff, spacing, maxDot);

    // Subtractive recombination on white paper, K overprints the others.
    vec3 col = vec3((1.0 - c) * (1.0 - k),
                    (1.0 - m) * (1.0 - k),
                    (1.0 - y) * (1.0 - k));
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "spacing": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "screen frequency / dot grid spacing"},
    "max_dot": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "max dot diameter as fraction of spacing"},
    "angle_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "rotate all screens (0.5 = 0 deg)"}
}
    )