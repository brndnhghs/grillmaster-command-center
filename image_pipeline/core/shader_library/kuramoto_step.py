"""kuramoto_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("kuramoto_step",
          "Kuramoto one Euler step: θ += dt·(Ω + K·Σsin(θⱼ−θ) + gK·R·sin(Ψ−θ))",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float theta = s.r;
    float omega = s.g;
    float rng = s.b;
    // Nearest-neighbour coupling term Σⱼ sin(θⱼ − θᵢ) (toroidal wrap).
    float cl = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float cr = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float cu = texture(u_texture, v_uv + vec2(0.0,  texel.y)).r;
    float cd = texture(u_texture, v_uv + vec2(0.0, -texel.y)).r;
    float coupling = sin(cl - theta) + sin(cr - theta)
                   + sin(cu - theta) + sin(cd - theta);
    // Global mean-field term: approximate R·sin(Ψ−θᵢ) with a frame-stable
    // proxy using the local-averaged phase (keeps the twin lively without a
    // full-canvas reduction in GLSL). Ψ ≈ neighbourhood mean phase.
    float neigh = (cl + cr + cu + cd) * 0.25;
    float mean_sin = sin(neigh - theta);
    float K  = u_params.x;            // local coupling
    float gK = u_params.y;            // global coupling
    float dt = max(u_params.w, 0.02);
    // local coherence ~ |coupling|/4 in [0,1] stands in for R so the global
    // term stays bounded and the pattern still forms chimeras.
    float Rloc = clamp(abs(coupling) / 4.0, 0.0, 1.0);
    float dtheta = omega + K * coupling + gK * Rloc * mean_sin;
    float ntheta = mod(theta + dt * dtheta, 6.2831853);
    // advance RNG carry
    float nrng = fract(rng * 1.4567 + 0.137);
    f_color = vec4(ntheta, omega, nrng, 1.0);
}
''')