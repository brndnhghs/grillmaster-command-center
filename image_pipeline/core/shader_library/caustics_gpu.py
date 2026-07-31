"""caustics_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Water Caustics (node 312) client-GPU twin ──
# Closed-form Jacobian caustics, same family as 125/164/172/53/43/57: every
# frame is a pure function of (uv, t) -> exact parity preview, no seeded-layout
# divergence. The CPU numpy node stays authoritative for export (two-tier
# precision). Analytic plane-wave derivatives keep it exact and cheap.
# Real params bound via CLIENT_GPU_SHIMS param_map (scale/gain/sharpen/speed);
# default colormode (ocean) is reproduced inline. pitfall #15: 0.5 -> node
# default so the neutral u_params yields the canonical view.
_register("caustics_gpu",
          "Water caustics — Jacobian light-convergence (client-GPU twin of node 312)",
          "procedural", '''\nvec3 caustics_ocean(float c, float floor) {
    // matches CPU colormode='ocean': deep blue floor, bright cyan caustic lines
    vec3 col = vec3(c * 0.35 + 0.02 + floor,
                    c * 0.85 + 0.12 + floor,
                    c * 0.75 + 0.30 + floor);
    return clamp(col, 0.0, 1.0);
}
void main() {
    // Decode real node params (0.5-neutral -> node defaults):
    //   p1 scale [1,16] 0.5->6.0 ; p2 caustic_gain [0.1,3] 0.5->1.2
    //   p3 sharpen [0.5,4] 0.5->1.6 ; p4 anim_speed [0.1,3] 0.5->1.0
    float scale = 0.5 + u_scale * 11.0;
    float gain  = 0.1 + u_caustic_gain * 2.2;
    float sharp = 0.5 + u_sharpen * 2.2;
    float aspd  = 0.1 + u_anim_speed * 1.8;
    float pht   = u_time * aspd;            // flow-mode phase advance

    // Pixel-centered, scale-normalized cartesian coords.
    vec2 res = u_resolution;
    vec2 pc = (gl_FragCoord.xy - 0.5 * res) / max(res.x, res.y) * scale;

    const int NW = 6;
    float H = 0.0, Hx = 0.0, Hy = 0.0;
    float Hxx = 0.0, Hxy = 0.0, Hyy = 0.0;
    for (int i = 0; i < NW; i++) {
        float fi = float(i);
        float ang = hash21(vec2(fi, 1.7)) * 6.2831853;
        float dx = cos(ang), dy = sin(ang);
        float k  = (0.8 + hash21(vec2(fi, 9.1)) * 1.6) * scale * 0.25;
        float ph = hash21(vec2(fi, 4.3)) * 6.2831853 + pht;
        float u  = k * (pc.x * dx + pc.y * dy) + ph;
        float A  = 1.0 / float(NW);          // fixed normalizer (amplitude in play)
        float s  = sin(u), co = cos(u);
        H   += A * s;
        Hx  += A * k * dx * co;
        Hy  += A * k * dy * co;
        Hxx += -A * k * k * dx * dx * s;
        Hxy += -A * k * k * dx * dy * s;
        Hyy += -A * k * k * dy * dy * s;
    }

    // Jacobian of displacement map W = (x + g*Hx, y + g*Hy).
    float j11 = 1.0 + gain * Hxx;
    float j12 = gain * Hxy;
    float j21 = gain * Hxy;
    float j22 = 1.0 + gain * Hyy;
    float det = j11 * j22 - j12 * j21;

    // Light convergence ~ 1/|det|; diverging (det>1) darkens the floor.
    float inv = max(abs(det), 1e-3);
    float caustic = 1.0 / inv;
    caustic = clamp(caustic, 0.0, 4.0);
    caustic = (det > 1.0) ? caustic * 0.25 : caustic;
    float c = clamp(pow(caustic / 4.0, sharp), 0.0, 1.0);

    // Dappled floor noise.
    float floor_ = (0.5 + 0.5 * fbm(pc * 2.0 + vec2(pht * 0.1))) * 0.15;
    vec3 col = caustics_ocean(c, floor_);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "caustic scale"},
    "caustic_gain": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "caustic gain"},
    "sharpen": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "sharpen exponent"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "flow speed"}
}
    )