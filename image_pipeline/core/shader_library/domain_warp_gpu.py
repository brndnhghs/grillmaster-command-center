"""domain_warp_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _inferno_local




# ── Domain Warping (node 311) client-GPU twin ──
# Inigo Quilez two-level domain warp: fbm noise displaced by a lower-frequency
# copy of itself. Every frame is a pure closed-form function of (uv, t) — same
# family as 125/164/172/53/43/57/312. The CPU numpy node stays authoritative for
# export (two-tier precision). Real params bound via CLIENT_GPU_SHIMS param_map
# (scale/warp_strength/contrast/octaves); colormode defaults to the node's
# 'inferno'. pitfall #15: 0.5 -> node default so neutral u_params yields the
# canonical marbled view. pitfall #19: amplitude is divided by a FIXED normalizer
# (sum of octave amps), so warp_strength stays live and is not silently cancelled.
_register("domain_warp_gpu",
          "Domain Warping — IQ two-level fractal noise warp (client-GPU twin of node 311)",
          "procedural", _inferno_local('') + '''
float dw_fbm(vec2 p, int oct) {
    float v = 0.0, a = 0.5, norm = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= oct) break;
        v += a * noise(p);
        norm += a;
        p *= 2.0; a *= 0.5;
    }
    return v / max(norm, 1e-6) * 2.0 - 1.0;   // [-1,1]
}
void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,12] 0.5->4.0 ; p2 warp_strength [0,8] 0.5->4.0
    //   p3 contrast [0.5,3] 0.5->1.25 ; p4 octaves [1,8] 0.5->4
    float scale = 1.0 + u_scale * 6.0;
    float warp  = u_warp_strength * 8.0;
    float contr = 0.5 + u_contrast * 1.5;
    int oct = int(clamp(1.0 + u_octaves * 7.0, 1.0, 8.0));

    vec2 p = (v_uv - 0.5) * scale;
    // Gentle time drift -> live preview evolves (canonical view at t=0).
    vec2 ph = vec2(u_time * 0.12);

    vec2 q = vec2(dw_fbm(p, oct),
                  dw_fbm(p + vec2(5.2, 1.3), oct));
    vec2 r = vec2(dw_fbm(p + warp * q + vec2(1.7, 9.2) + ph, oct),
                  dw_fbm(p + warp * q + vec2(8.3, 2.8) + ph, oct));
    float val = dw_fbm(p + warp * r, oct);

    val = (val + 1.0) * 0.5;
    val = clamp(0.5 + (val - 0.5) * contr, 0.0, 1.0);

    vec3 col = inferno(val);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "noise scale"},
    "warp_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "warp strength"},
    "contrast": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "contrast"},
    "octaves": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0, "description": "fbm octaves"}
}
    )