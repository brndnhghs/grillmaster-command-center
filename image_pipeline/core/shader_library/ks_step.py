"""ks_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ks_step",
          "Kuramoto-Sivashinsky one step: -nu*∇⁴u - ∇²u - ½|∇u|² (5-pt operators + hashed noise)",
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
    // 5-pt biharmonic (∇⁴) from the Laplacian
    float l2 = texture(u_texture, v_uv + vec2(-texel.x*2.0, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x*2.0, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y*2.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y*2.0)).r
             - 4.0 * ((l+r+d+uu)*0.25);
    float lap2 = lap + (l2 - lap) * 0.5;  // cheap ∇⁴ stand-in
    // gradient magnitude squared |∇u|²
    float dx = (r - l) * 0.5, dy = (uu - d) * 0.5;
    float grad2 = dx*dx + dy*dy;
    float nu = clamp(u_params.x, 0.01, 0.5);
    float dt = clamp(u_params.y, 0.001, 0.05);
    float sigma = clamp(u_params.z, 0.0, 0.3);
    phase = fract(phase + dt);
    float eta = (hash21(floor(v_uv * u_resolution) + phase * 211.0) - 0.5) * sigma;
    float un = c + dt * (-nu * lap2 - lap - 0.5 * grad2 + eta);
    float peak = max(abs(un), 1.0);
    if (peak > 6.0) { un *= 6.0 / peak; }
    f_color = vec4(clamp(un, -6.0, 6.0), phase, 0.0, 1.0);
}
''')