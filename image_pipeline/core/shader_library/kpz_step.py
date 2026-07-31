"""kpz_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("kpz_step",
          "KPZ one step (diffusion + nonlinear growth + white-noise source)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float h = s.r, phase = s.b;
    float lh = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * h;
    float nu = clamp(u_params.x, 0.05, 3.0);
    float lam = clamp(u_params.y, -3.0, 3.0);
    float sigma = clamp(u_params.z, 0.01, 1.0);
    float dt = clamp(u_params.w, 0.01, 1.0);
    // Gradient magnitude squared |∇h|²
    float dx = (texture(u_texture, v_uv + vec2(texel.x,0.0)).r
                - texture(u_texture, v_uv + vec2(-texel.x,0.0)).r) * 0.5;
    float dy = (texture(u_texture, v_uv + vec2(0.0,texel.y)).r
                - texture(u_texture, v_uv + vec2(0.0,-texel.y)).r) * 0.5;
    float grad2 = dx*dx + dy*dy;
    // White-noise source (hashed, time-evolving)
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 137.0) - 0.5);
    float dh = nu * lh + 0.5 * lam * grad2 + sigma * eta * 2.0;
    float hn = h + dt * dh;
    float peak = max(abs(hn), 1.0);
    if (peak > 10.0) { hn *= 10.0 / peak; }
    f_color = vec4(clamp(hn, -10.0, 10.0), phase, 0.0, 1.0);
}
''')