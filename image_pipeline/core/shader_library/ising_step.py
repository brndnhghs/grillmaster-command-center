"""ising_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ising_step",
          "Ising one Glauber step: flip spin by Metropolis-like prob from 4-neighbour sum (node 93 twin)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float J = clamp(u_params.x, 0.5, 2.0);
    float T = clamp(u_params.y, 0.5, 3.0);
    float beta = 1.0 / (T * 2.2691853);     // Tc(Tc-scaled) = 2.269*J, J folded out
    float spin = s.r > 0.5 ? 1.0 : -1.0;
    float rng = fract(s.b * 1.824 + 0.317);

    // 4-neighbour von Neumann sum (+1/-1 each)
    float nsum = 0.0;
    nsum += (texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r > 0.5 ? 1.0 : -1.0);
    nsum += (texture(u_texture, v_uv + vec2( texel.x, 0.0)).r > 0.5 ? 1.0 : -1.0);
    nsum += (texture(u_texture, v_uv + vec2(0.0, texel.y)).r > 0.5 ? 1.0 : -1.0);
    nsum += (texture(u_texture, v_uv + vec2(0.0,-texel.y)).r > 0.5 ? 1.0 : -1.0);

    float dE = 2.0 * J * spin * nsum;        // energy cost of flipping
    float p_flip = dE > 0.0 ? exp(-beta * dE) : 1.0;
    float nspin = (rng < p_flip) ? -spin : spin;
    f_color = vec4(nspin > 0.5 ? 1.0 : 0.0, 0.0, rng, 1.0);
}
''')