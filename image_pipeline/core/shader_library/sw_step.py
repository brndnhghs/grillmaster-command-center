"""sw_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sw_step",
          "Shallow Water one step (wave-coupled height/velocity, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float h = s.r, u = s.g, v = s.b;
    float lh = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * h;
    float g = clamp(u_params.x, 1.0, 20.0);
    float h0 = clamp(u_params.y, 0.3, 3.0);
    float nu = clamp(u_params.z, 0.0001, 0.005);
    float amp = clamp(u_params.w, 0.02, 0.5);
    float dt = 0.08;
    // Gravity-driven height gradient -> velocity (wave coupling)
    float du = -g * (texture(u_texture, v_uv + vec2(texel.x,0.0)).r
                     - texture(u_texture, v_uv + vec2(-texel.x,0.0)).r) * 0.5;
    float dv = -g * (texture(u_texture, v_uv + vec2(0.0,texel.y)).r
                     - texture(u_texture, v_uv + vec2(0.0,-texel.y)).r) * 0.5;
    // Viscosity smooths velocity
    u += (nu * lh * 30.0 + du) * dt;
    v += (nu * lh * 30.0 + dv) * dt;
    // Height advected by velocity + diffusion
    float hn = h + dt * (0.20 * lh - (u + v) * 0.5);
    // Central ripple source (mirrors CPU two-point source)
    vec2 p0 = vec2(0.33, 0.5);
    float ds = distance(v_uv, p0);
    hn += amp * sin(u_time * 3.0) * exp(-(ds*ds)/0.002);
    float peak = max(abs(hn), 1.0);
    if (peak > 6.0) { hn *= 6.0 / peak; u *= 6.0 / peak; v *= 6.0 / peak; }
    f_color = vec4(clamp(hn, -6.0, 6.0), clamp(u, -6.0, 6.0), clamp(v, -6.0, 6.0), 1.0);
}
''')