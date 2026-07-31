"""spd125_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spd125_step",
          "SPD #153 one Fermi-imitation step: probabilistic strategy switch from neighbors (snowdrift matrix)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float my = s.r;
    float rng = fract(s.b * 1.4567 + 0.137);

    // Moore neighborhood (8 cells)
    float n00 = texture(u_texture, v_uv + vec2(-texel.x,-texel.y)).r;
    float n01 = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float n02 = texture(u_texture, v_uv + vec2(texel.x,-texel.y)).r;
    float n10 = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r;
    float n12 = texture(u_texture, v_uv + vec2(texel.x,0.0)).r;
    float n20 = texture(u_texture, v_uv + vec2(-texel.x,texel.y)).r;
    float n21 = texture(u_texture, v_uv + vec2(0.0,texel.y)).r;
    float n22 = texture(u_texture, v_uv + vec2(texel.x,texel.y)).r;
    float nc = n00+n01+n02+n10+n12+n20+n21+n22;  // #coop neighbors
    float nn = 8.0 - nc;                         // #defect neighbors

    float T = clamp(u_params.x, 1.0, 2.0);     // temptation payoff
    float S = clamp(u_params.y, -1.0, 1.0);    // sucker payoff
    float K = clamp(u_params.z, 0.01, 2.0);    // Fermi stochasticity

    // Snowdrift payoffs: coop reward R=1.0, defect gets T vs coop / S vs defect
    float my_pay  = (my < 0.5) ? (nc * 1.0 + nn * S) : (nc * T);
    float r2 = fract(rng * 2.137 + 0.71);
    float nbr = step(0.5, r2);                  // a random neighbor strategy
    float nbr_pay = (nbr < 0.5) ? (nc * 1.0 + nn * S) : (nc * T);

    float prob = 1.0 / (1.0 + exp((my_pay - nbr_pay) / K));
    float r3 = fract(rng * 3.11 + 0.43);
    float nstrat = (r3 < prob) ? nbr : my;

    // rough normalized payoff for display brightness
    float pay = clamp(my_pay / (8.0 * max(T, 1.0)), 0.0, 1.0);
    f_color = vec4(nstrat, pay, rng, 1.0);
}
''')