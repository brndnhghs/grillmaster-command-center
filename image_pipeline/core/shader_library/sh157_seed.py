"""sh157_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 157: Swift-Hohenberg (r·u − (∇²+q₀²)²·u − u³) ─────────────────────
# Same ε·u − u³ structure with a tuned wavenumber band. q0 packs into p2.
# p1=r, p2=q0, p3=dt, p4=noise_amp.
_register("sh157_seed",
          "Swift-Hohenberg (157) seed: small hashed field + q0-scale hex hint (node 157 twin)",
          "procedural", '''
void main() {
    float q0 = clamp(u_params.y, 0.02, 0.3);
    float u = 0.15 * (cos(q0 * v_uv.x * 6.2831853)
                      + cos(q0 * 0.5 * v_uv.x * 6.2831853 + 0.8660254 * q0 * v_uv.y * 6.2831853)
                      + cos(q0 * 0.5 * v_uv.x * 6.2831853 - 0.8660254 * q0 * v_uv.y * 6.2831853));
    u += 0.05 * (noise(v_uv * 7.0) - 0.5);
    f_color = vec4(u, 0.0, 0.0, 1.0);  // R=u, G=phase
}
''')