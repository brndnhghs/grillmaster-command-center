"""frac_rd_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.5 — Fractional Laplacian Reaction-Diffusion (node 163) RD twin
# ═══════════════════════════════════════════════════════════════════════════
# Gray-Scott reaction (identical kinetics: −UV²+F(1−U) / +UV²−(F+k)V) with the
# fractional exponent α approximated LOCALLY as an effective substrate-diffusion
# breadth (lower α → broader Du → coarser/more-replicating spots). The true
# Fourier (-∇²)^(α/2) operator is not reproduced — the CPU numpy node stays the
# authoritative export. Seed reuses grayscott_seed; display uses the node's fire
# colormap. State: U in .r, V in .g.
_register("frac_rd_step",
          "Fractional-RD step: Gray-Scott reaction + α-modulated diffusion breadth (node 163)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 st = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sb = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + st.r + sb.r - 4.0 * U;
    float lapV = sl.g + sr.g + st.g + sb.g - 4.0 * V;
    float F     = clamp(u_params.x, 0.01, 0.08);
    float k     = clamp(u_params.y, 0.03, 0.08);
    float alpha = clamp(u_params.z, 0.5, 2.0);
    float Dv    = clamp(u_params.w, 0.005, 0.2);
    // Lévy-flight proxy: lower α → broader substrate diffusion (sharper fronts,
    // more self-replication). α=2 → Du=1.5·Dv; α=0.5 → Du≈3.75·Dv.
    float Du = Dv * (1.5 + (2.0 - alpha) * 1.5);
    float uvv = U * V * V;
    float nU = clamp(U + (Du * lapU - uvv + F * (1.0 - U)), 0.0, 1.0);
    float nV = clamp(V + (Dv * lapV + uvv - (F + k) * V), 0.0, 1.0);
    f_color = vec4(nU, nV, 0.0, 1.0);
}
''')