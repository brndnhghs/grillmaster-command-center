"""cahn_hilliard_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cahn_hilliard_step",
          "Cahn-Hilliard one step (5-pt Laplacian, toroidal) — two-channel state (.r=φ, .g=μ)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float phi = s.r;
    float mu  = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float eps = max(u_params.x, 0.01);   // interface width (p1)
    float mob = max(u_params.y, 0.01);   // mobility (p2)
    // Stable explicit dt for Model B: dt < 2/(mob*eps^2*kmax^2); kmax^2~9.87
    float dt = min(0.05, 1.5 / (mob * eps * eps * 9.87 + 1e-3));
    float lap_phi = sl.r + sr.r + su.r + sd.r - 4.0 * phi;
    float mu_new  = phi * phi * phi - phi - eps * eps * lap_phi;
    float lap_mu  = sl.g + sr.g + su.g + sd.g - 4.0 * mu;
    float phi_new = phi + dt * lap_mu;
    f_color = vec4(clamp(phi_new, -1.5, 1.5), mu_new, 0.0, 1.0);
}
''')