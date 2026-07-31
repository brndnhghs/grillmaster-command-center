"""pfc_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("pfc_step",
          "Phase Field Crystal one step: lap(psi + lap psi) - r/2 psi^2 + psi^3 + noise",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float psi = s.r, phase = s.g;
    float c  = psi;
    float l  = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float r  = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float d  = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float uu = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float lap = (l + r + d + uu - 4.0 * c);
    float lap2 = (l + r + d + uu - 4.0 * lap) - 4.0 * c;  // ∇⁴ approx
    float eps = clamp(u_params.x, 0.01, 0.5);
    float dt = clamp(u_params.y, 0.01, 0.5);
    float sigma = clamp(u_params.z, 0.0, 0.1);
    float r2 = clamp(u_params.w, 0.0, 2.0);  // = r/2 quadratic coefficient
    float lin = lap + lap2;
    float reaction = eps * c - r2 * c * c + (1.0 / 3.0) * c * c * c;
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 31.0) - 0.5) * sigma * 12.0;
    float pn = c + dt * (lin + reaction + eta);
    float peak = max(abs(pn), 1.0);
    if (peak > 4.0) { pn *= 4.0 / peak; }
    f_color = vec4(clamp(pn, -4.0, 4.0), phase, 0.0, 1.0);
}
''')