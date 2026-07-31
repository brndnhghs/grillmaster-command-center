"""nematic_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("nematic_step",
          "Active-nematic step: Landau-de Gennes + activity + elastic ∇²Q + hash noise (node 99)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s  = texture(u_texture, v_uv);
    float Qxx = s.r, Qxy = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 st = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sb = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float Lxx = sl.r + sr.r + st.r + sb.r - 4.0 * Qxx;
    float Lxy = sl.g + sr.g + st.g + sb.g - 4.0 * Qxy;
    float alpha = clamp(u_params.x, -0.2, 0.2);      // activity α
    float D     = clamp(u_params.y, 0.01, 2.0);      // elastic constant
    float A     = clamp(u_params.z, -0.5, 0.1);      // Landau A
    float noise = clamp(u_params.w, 0.0, 0.15);      // thermal noise amplitude
    const float C = 1.0, G = 1.0, dt = 0.05;
    float S2 = 2.0 * (Qxx * Qxx + Qxy * Qxy);        // Tr(Q²)
    float Hxx = -(A * Qxx + C * S2 * Qxx);
    float Hxy = -(A * Qxy + C * S2 * Qxy);
    float dQxx = dt * (G * Hxx + alpha * Qxx + D * Lxx);
    float dQxy = dt * (G * Hxy + alpha * Qxy + D * Lxy);
    // Thermal noise (nucleates defects) — state-dependent so it varies per substep.
    float n1 = hash21(v_uv * 512.0 + vec2(Qxx, Qxy) * 813.0 + 2.3) - 0.5;
    float n2 = hash21(v_uv * 727.0 + vec2(Qxy, Qxx) * 611.0 + 7.1) - 0.5;
    dQxx += noise * n1 * 2.0 * 0.2236;               // 2·(hash−0.5)·√dt
    dQxy += noise * n2 * 2.0 * 0.2236;
    f_color = vec4(clamp(Qxx + dQxx, -2.0, 2.0),
                   clamp(Qxy + dQxy, -2.0, 2.0), 0.0, 1.0);
}
''')