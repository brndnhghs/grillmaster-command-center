"""sh128_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sh128_step",
          "Swift-Hohenberg (128) one step: epsilon*u - u^3 - (1+lap)^2 u + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, phase = s.g;
    // 5-pt Laplacian
    float c  = u;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    // (1 + lap)^2 u  = u + 2*lap + lap*lagain  (local ∇⁴ stand-in)
    float lap2 = (l + r + d + uu - 4.0 * lap) - 4.0 * c;  // ∇⁴ approx
    float lap4 = lap + lap2;
    float eps = clamp(u_params.x, -0.5, 3.0);
    float dt = clamp(u_params.y, 0.01, 1.0);
    float sigma = clamp(u_params.z, 0.0, 1.0);
    float gain = clamp(u_params.w, 0.0, 2.0);
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 97.0) - 0.5) * sigma;
    float reaction = eps * c - c * c * c;
    // -(1 + gain*lap)^2 u  ~  -u - 2*gain*lap - gain*gain*lap2
    float lin = -c - 2.0 * gain * lap - gain * gain * lap2;
    float un = c + dt * (reaction + lin + eta);
    float peak = max(abs(un), 1.0);
    if (peak > 4.0) { un *= 4.0 / peak; }
    f_color = vec4(clamp(un, -4.0, 4.0), phase, 0.0, 1.0);
}
''')