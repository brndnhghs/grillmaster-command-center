"""caustics513_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Caustics client-GPU twin (node 513) ───────────────────────────────────
# Closed-form f(uv, t) preview of GPU-Gems water caustics (Guardado &
# Sánchez-Crespo 2003). The CPU node 513 computes 1/|det J| of the floor->
# refracted-landing map via a finite-difference Jacobian; the Hessian of the
# sum-of-sines height field is available analytically (second derivative of a
# sine is -k^2 sin), so the same inverse-area-magnification caustic is a pure
# function of (uv, t) here — exact-family parity preview, no seeded-layout
# divergence. Wave directions/freqs are fixed (seed-independent live preview);
# the CPU numpy node stays the authoritative export (two-tier precision).
_register("caustics513_gpu",
          "Water caustics inverse-magnification field (client-GPU twin of node 513)",
          "procedural", '''
void main() {
    // u_depth (0.2..3), u_scale (1..20), u_gain (0.5..8), u_waves (2..7)
    // map directly to the node's real params (typed-uniform live path).
    float depth = u_depth;
    float scl   = u_scale;
    float gain  = u_gain;
    int   nw    = int(u_waves + 0.5);
    float t     = u_time * 0.6;
    vec2  uv    = (v_uv - 0.5) * 2.0;   // [-1, 1] floor coords

    // Analytic Hessian of the sum-of-sines wave height H(u,v,t).
    float Hxx = 0.0, Hyy = 0.0, Hxy = 0.0;
    for (int i = 0; i < 7; i++) {
        if (i >= nw) break;
        float fi   = float(i);
        float ang  = fi * 1.7 + 0.5;                 // fixed directions
        float freq = (0.7 + 0.22 * fi) * scl;
        float amp  = 0.8 / (1.0 + 0.3 * fi);
        float sp   = 0.6 + 0.2 * fi;
        vec2  k    = freq * vec2(cos(ang), sin(ang));
        float ph   = dot(k, uv) + sp * t + fi * 1.3;
        float s    = sin(ph);
        Hxx += -amp * k.x * k.x * s;
        Hyy += -amp * k.y * k.y * s;
        Hxy += -amp * k.x * k.y * s;
    }

    // Jacobian of floor-point -> refracted-landing map ~ I + depth * Hessian.
    float Jxx  = 1.0 + depth * Hxx;
    float Jyy  = 1.0 + depth * Hyy;
    float Jxy  = depth * Hxy;
    float detJ = Jxx * Jyy - Jxy * Jxy;
    float mag  = abs(detJ);

    // Caustic intensity: inverse area magnification, baseline-subtracted then
    // soft-saturated + perceptual falloff (matches the CPU node's shaping).
    float caustic = max(1.0 / max(mag, 1e-3) - 1.0, 0.0);
    caustic = caustic / (caustic + 1.0);
    caustic = pow(caustic, 0.7);
    caustic = clamp(caustic * gain * 0.5, 0.0, 1.0);

    // Deep-water floor + aqua-tinted filaments (default colormode).
    vec3 floorc = vec3(0.02, 0.10, 0.16);
    vec3 tint   = vec3(0.55, 0.95, 1.0);
    vec3 col    = floorc + caustic * tint * 1.4;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "depth": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0, "description": "water-column depth (refraction focusing)"},
    "scale": {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0, "description": "surface wave spatial frequency"},
    "gain":  {"glsl": "float", "min": 0.5, "max": 8.0, "default": 3.0, "description": "caustic brightness gain"},
    "waves": {"glsl": "float", "min": 2.0, "max": 7.0, "default": 4.0, "description": "number of superimposed directional waves"}
}
    )