"""wave_eq_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("wave_eq_step",
          "Wave Equation one step (velocity-Verlet, 5-pt toroidal Laplacian)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g, phase = s.b;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float c = clamp(u_params.x, 0.3, 2.5);
    float c2 = 0.20 * c * c;             // stable ( < 0.5 )
    float damp = clamp(u_params.y, 0.90, 1.0);
    float freq = clamp(u_params.z, 0.2, 8.0);
    float amp = clamp(u_params.w, 0.2, 5.0);

    // Source injection: two detuned point sources (mirrors the CPU node).
    float dphi = 6.2831853 * freq;
    phase = mod(phase + dphi, 6.2831853);
    float src = amp * sin(phase);
    vec2 p0 = vec2(0.33, 0.5), p1v = vec2(0.66, 0.5);
    float ds0 = distance(v_uv, p0), ds1 = distance(v_uv, p1v);
    float src_inj = src * (exp(-(ds0*ds0)/0.0008) + exp(-(ds1*ds1)/0.0008) * 0.85);

    float vn = (v + c2 * lu) * damp;       // dv = c2*lap ; velocity damping
    float un = u + vn + src_inj;           // du = v
    f_color = vec4(clamp(un, -8.0, 8.0), clamp(vn, -8.0, 8.0), phase, 1.0);
}
''')