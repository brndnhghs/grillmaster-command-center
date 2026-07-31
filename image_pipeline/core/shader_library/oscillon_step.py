"""oscillon_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("oscillon_step",
          "Oscillon Resonance one step (parametric Mathieu pump, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g, phase = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float eps = clamp(u_params.x, 0.05, 1.5);
    float w0 = clamp(u_params.y, 0.5, 6.0);
    float gamma = clamp(u_params.z, 0.01, 1.0);
    float D = clamp(u_params.w, 0.05, 4.0);
    float dt = 0.1;
    float pump = 2.0 * w0;                   // omega_p = 2*omega0
    phase = mod(phase + dt * pump, 6.2831853);
    float stiff = w0 * w0 * (1.0 + eps * sin(phase));
    float beta = 0.3;
    float force = D * lu - gamma * v - stiff * u - beta * u * u * u;
    float vn = v + dt * force;
    float un = u + dt * vn;
    float peak = max(abs(un), 1.0);
    if (peak > 8.0) { un *= 8.0 / peak; vn *= 8.0 / peak; }
    f_color = vec4(un, vn, phase, 1.0);
}
''')