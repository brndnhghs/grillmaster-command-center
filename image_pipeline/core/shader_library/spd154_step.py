"""spd154_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spd154_step",
          "CSPD #154 one Euler step: replicator reaction + diffusion + mutation drift + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float s0 = s.r;
    float accum = s.g;
    float rng = fract(s.b * 1.245 + 0.371);

    float T = u_params.x;   // temptation
    float R = u_params.y;   // reward (mutual coop)
    float S = u_params.z;   // sucker
    float P = u_params.w;   // punishment

    // 8-neighbour sum for replicator payoff fields
    float sum_s = 0.0;
    sum_s += texture(u_texture, v_uv + vec2(-texel.x,-texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(texel.x,-texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(-texel.x,0.0)).r;
    sum_s += texture(u_texture, v_uv + vec2(texel.x,0.0)).r;
    sum_s += texture(u_texture, v_uv + vec2(-texel.x,texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(0.0,texel.y)).r;
    sum_s += texture(u_texture, v_uv + vec2(texel.x,texel.y)).r;

    float coop_sum = R * sum_s + S * (8.0 - sum_s);
    float def_sum  = T * sum_s + P * (8.0 - sum_s);
    float replicator = s0 * (1.0 - s0) * (coop_sum - def_sum);

    float mutation = 0.025;
    float mutation_drift = mutation * (0.5 - s0);

    // 5-point Laplacian for diffusion
    float lap = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r
              + texture(u_texture, v_uv + vec2(texel.x,0.0)).r
              + texture(u_texture, v_uv + vec2(0.0,texel.y)).r
              + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r
              - 4.0 * s0;
    float D = 0.12;
    float diffusion = D * lap;

    // cheap pseudo-Gaussian noise (sum of 3 uniforms, variance-normalized)
    float gnoise = (fract(rng * 1.7 + 0.13) + fract(rng * 2.3 + 0.57)
                    + fract(rng * 3.1 + 0.91) - 1.5) * 0.5773502;
    float noise_amp = 0.008;
    float DT = 0.2;
    float ds = DT * (replicator + mutation_drift + diffusion)
               + noise_amp * gnoise * sqrt(DT);
    float sn = clamp(s0 + ds, 0.0, 1.0);

    float decay = 0.9;
    float an = decay * accum + (1.0 - decay) * sn;
    f_color = vec4(sn, clamp(an, 0.0, 1.0), rng, 1.0);
}
''')