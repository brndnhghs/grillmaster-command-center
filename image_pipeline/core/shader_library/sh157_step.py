"""sh157_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sh157_step",
          "Swift-Hohenberg (157) one step: r*u - (lap+q0^2)^2 u - u^3 + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, phase = s.g;
    float c  = u;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    float lap2 = (l + r + d + uu - 4.0 * lap) - 4.0 * c;  // ∇⁴ approx
    float q0 = clamp(u_params.y, 0.02, 0.3);
    float rq02 = (lap + q0 * q0) * (lap + q0 * q0);  // (∇²+q0²) for lin term
    float rq04 = rq02 * rq02;                          // (∇²+q0²)²
    float rr = clamp(u_params.x, -1.0, 5.0);
    float dt = clamp(u_params.z, 0.01, 0.5);
    float sigma = clamp(u_params.w, 0.0, 0.1);
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 53.0) - 0.5) * sigma * 10.0;
    float reaction = rr * c - c * c * c;
    float lin = -rq04;
    float un = c + dt * (reaction + lin + eta);
    float peak = max(abs(un), 1.0);
    if (peak > 4.0) { un *= 4.0 / peak; }
    f_color = vec4(clamp(un, -4.0, 4.0), phase, 0.0, 1.0);
}
''')