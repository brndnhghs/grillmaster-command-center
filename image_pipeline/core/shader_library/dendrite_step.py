"""dendrite_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dendrite_step",
          "Dendritic Allen-Cahn step: anisotropic W(θ)²∇²φ + double-well + thermal (node 122)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float phi = s.r, uu = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 st = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sb = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapPhi = sl.r + sr.r + st.r + sb.r - 4.0 * phi;
    float lapU   = sl.g + sr.g + st.g + sb.g - 4.0 * uu;
    float phix = (sr.r - sl.r) * 0.5;
    float phiy = (st.r - sb.r) * 0.5;
    float theta = atan(phiy, phix);
    float eps = clamp(u_params.y, 0.0, 0.1);                 // anisotropy
    float k   = floor(clamp(u_params.z, 3.0, 8.0) + 0.5);    // symmetry (int fold)
    float W0 = 0.5;
    float w  = W0 * (1.0 + eps * cos(k * theta));
    float aniso = (w * w) * lapPhi;
    float fp    = 4.0 * phi * (phi * phi - 1.0);             // f'(φ)=(φ²−1)²'
    float drive = 4.0 * (1.0 - phi * phi);                   // D_DRIVE=4
    float M  = 50.0;
    // Cap the step for explicit-scheme stability (M·dt·w² must stay <~0.25, else
    // the interior checkerboards). The CPU node masks this with a periodic
    // Gaussian blur; the twin caps dt + lightly smooths in display instead.
    float dt = min(clamp(u_params.w, 0.005, 0.2), 0.02);
    float dphi = M * (aniso - fp + drive);
    float phiN = clamp(phi + dt * dphi, -1.0, 1.0);
    float du   = 6.0 * lapU + 0.3 * max(dphi, 0.0);          // D_THERMAL=6 + latent heat
    float uN   = clamp(uu + dt * du, -1.0, 1.0);
    f_color = vec4(phiN, uN, 0.0, 1.0);
}
''')