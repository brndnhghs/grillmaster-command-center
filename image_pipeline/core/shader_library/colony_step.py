"""colony_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("colony_step",
          "Bacterial colony step: N nutrient, C colony (5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float N = s.r, C = s.g;
    float ln = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0*N;
    float lc = texture(u_texture, v_uv + vec2(-texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(texel.x,0.0)).g
             + texture(u_texture, v_uv + vec2(0.0,texel.y)).g
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).g - 4.0*C;
    float growth = u_params.x, diff_c = u_params.y, cons = u_params.z, death = u_params.w;
    float dC = growth * C * N - cons * C + diff_c * lc;
    float dN = -cons * C * N + 0.05 * ln;   // nutrient consumed + diffuses in
    float nC = clamp(C + 0.1 * dC, 0.0, 1.0);
    float nN = clamp(N + 0.1 * dN, 0.0, 1.0);
    f_color = vec4(nN, nC, 0.0, 1.0);
}
''')