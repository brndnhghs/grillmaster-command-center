"""curl_noise_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Curl-Noise Flow Field (node 314) client-GPU twin ──
# Divergence-free flow via curl of an fbm potential P: v = (dP/dy, -dP/dx).
# Velocity ANGLE -> hue, MAGNITUDE -> brightness (node default 'spectral' colormap).
# Closed-form function of (uv, t); CPU numpy node stays authoritative for export.
# Real params bound via CLIENT_GPU_SHIMS param_map (scale/octaves/brightness/
# anim_mode). pitfall #15: 0.5 -> node default. A subtle u_time pan keeps the
# live preview in motion (canonical view at t=0).
_register("curl_noise_gpu",
          "Curl-Noise flow field — divergence-free direction field (client-GPU twin of node 314)",
          "procedural", '''vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
float cn_fbm(vec2 p, int oct) {
    float v = 0.0, a = 0.5, norm = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= oct) break;
        // golden-angle rotate each octave so layers don't align on axes
        float ang = 2.3999632 * float(i + 1);
        vec2 rp = rot(ang) * p;
        v += a * noise(rp);
        norm += a;
        p *= 2.0; a *= 0.5;
    }
    return v / max(norm, 1e-6);
}
void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,12] 0.5->6.5 ; p2 octaves [1,6] 0.5->3
    //   p3 brightness [0.2,2] 0.5->1.1 ; p4 anim_mode [0,1] 0.5->drift
    float scale = 1.0 + u_scale * 11.0;
    int oct = int(clamp(1.0 + u_octaves * 5.0, 1.0, 6.0));
    float bright = 0.2 + u_brightness * 1.8;
    float drift = step(0.5, u_anim_mode);   // 0=static, 1=drift

    vec2 p = (v_uv - 0.5) * scale;
    vec2 pan = vec2(u_time * 0.6, u_time * 0.25) * drift;

    float e = 0.35;   // finite-difference step in noise space
    float P0 = cn_fbm(p + pan, oct);
    float Px = cn_fbm(p + pan + vec2(e, 0.0), oct);
    float Py = cn_fbm(p + pan + vec2(0.0, e), oct);
    float vx = (Py - P0) / e;        // dP/dy
    float vy = -(Px - P0) / e;       // -dP/dx
    float mag = length(vec2(vx, vy));
    float ang = atan(vy, vx);

    float hue = (ang + 3.14159265) / 6.2831853;
    float sat = clamp(0.5 + mag * 1.5, 0.0, 1.0);
    float val = clamp((0.25 + mag * 2.0) * bright, 0.0, 1.0);
    vec3 col = hsv2rgb(vec3(hue, sat, val));
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "noise scale"},
    "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 3.0, "description": "fbm octaves"},
    "brightness": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "brightness"},
    "anim_mode": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "drift on/off"}
}
    )