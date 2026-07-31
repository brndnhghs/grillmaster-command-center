"""acpm_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("acpm_step",
          "AC+PM step: Allen-Cahn reaction + Perona-Malik anisotropic diffusion (5-pt)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float c = s.r;
    float cl = texture(u_texture, v_uv + vec2(-texel.x,0.0)).r;
    float cr = texture(u_texture, v_uv + vec2(texel.x,0.0)).r;
    float cd = texture(u_texture, v_uv + vec2(0.0,texel.y)).r;
    float cu = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    // Perona-Malik anisotropic diffusion (4-neighbour, edge-preserving)
    float K2 = max(u_params.y * u_params.y, 1e-4);
    float gx = (cr - c) / (1.0 + (cr - c) * (cr - c) / K2);
    float gy = (cu - c) / (1.0 + (cu - c) * (cu - c) / K2);
    float gxl = (c - cl) / (1.0 + (c - cl) * (c - cl) / K2);
    float gyl = (c - cd) / (1.0 + (c - cd) * (c - cd) / K2);
    float diff = (gx - gxl) + (gy - gyl);
    // Allen-Cahn double-well reaction + constant bias
    float ac = c - c * c * c + u_params.z;
    float dt = u_params.w;
    float alpha = u_params.x;
    float nc = c + dt * (ac + alpha * diff);
    f_color = vec4(clamp(nc, -1.5, 1.5), 0.0, 0.0, 1.0);
}
''')