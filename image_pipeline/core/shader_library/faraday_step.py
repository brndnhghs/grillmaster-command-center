"""faraday_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("faraday_step",
          "Faraday Waves one step (parametric pump, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g, phase = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float A = u_params.x, w0 = clamp(u_params.y, 0.5, 6.0);
    float gamma = clamp(u_params.z, 0.02, 1.5);
    float nu = clamp(u_params.w, 0.05, 4.0);
    float dt = 0.08;
    float Omega = 2.0 * w0;                  // drive at 2*omega0 (subharmonic)
    phase = mod(phase + dt * Omega, 6.2831853);
    float drive = A * cos(phase);
    float alpha = 0.5;
    float force = nu * lu - gamma * v - (w0 * w0 + drive) * u + alpha * u * u * u;
    float vn = v + dt * force;
    float un = u + dt * vn;
    // soft clamp to avoid blowup
    float peak = max(abs(un), 1.0);
    if (peak > 8.0) { un *= 8.0 / peak; vn *= 8.0 / peak; }
    f_color = vec4(un, vn, phase, 1.0);
}
''')