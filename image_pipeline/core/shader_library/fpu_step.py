"""fpu_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("fpu_step",
          "FPU one step (Verlet, 5-pt nonlinear spring coupling)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g;
    float up = texture(u_texture, v_uv + vec2( texel.x, 0.0)).r;
    float um = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r;
    float vp = texture(u_texture, v_uv + vec2(0.0, texel.y)).r;
    float vm = texture(u_texture, v_uv + vec2(0.0,-texel.y)).r;
    float k2 = clamp(u_params.x, 0.1, 5.0);
    float k3 = clamp(u_params.y, 0.0, 2.0);
    float k4 = clamp(u_params.z, 0.0, 2.0);
    float dt = clamp(u_params.w, 0.01, 0.2);
    // Nonlinear spring force (discrete laplacian of force) — mirrors CPU acceleration()
    float dxp = up - u, dxm = u - um;
    float dyp = vp - u, dym = u - vm;
    float fx = k2 * (dxp - dxm) + k3 * (dxp*dxp - dxm*dxm) + k4 * (dxp*dxp*dxp - dxm*dxm*dxm);
    float fy = k2 * (dyp - dym) + k3 * (dyp*dyp - dym*dym) + k4 * (dyp*dyp*dyp - dym*dym*dym);
    float f = fx + fy;
    // Velocity-Verlet (per-frame v carried; u advanced by v + accel)
    float vn = v + f * dt;
    float un = u + vn * dt;
    float peak = max(abs(un), 1.0);
    if (peak > 30.0) { un *= 30.0 / peak; vn *= 30.0 / peak; }
    f_color = vec4(clamp(un, -30.0, 30.0), clamp(vn, -30.0, 30.0), 0.0, 1.0);
}
''')