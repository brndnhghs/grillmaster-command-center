"""nls_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("nls_step",
          "NLSE one Euler step (5-pt Laplacian, toroidal) — complex field in RG",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapR = sl.r + sr.r + su.r + sd.r - 4.0 * a;
    float lapI = sl.g + sr.g + su.g + sd.g - 4.0 * b;
    float beta = clamp(u_params.x, -2.0, 2.0);   // p1: node dispersion β
    float gnl  = clamp(u_params.y, -3.0, 3.0);   // p2: node nonlinearity g (+focus)
    float dt   = clamp(u_params.z, 0.002, 0.1);  // p3: node dt
    float trap = u_params.w;                      // p4: node trap_strength
    float r2 = (v_uv.x - 0.5) * (v_uv.x - 0.5)
             + (v_uv.y - 0.5) * (v_uv.y - 0.5);
    float V = trap * 400.0 * r2;                  // harmonic trap (live scale)
    float m = a * a + b * b;
    // ∂a/∂t = -β·lapI + g·m·b - V·b ;  ∂b/∂t = β·lapR - g·m·a + V·a
    float da = -beta * lapI + gnl * m * b - V * b;
    float db =  beta * lapR - gnl * m * a + V * a;
    float na = a + dt * da;
    float nb = b + dt * db;
    // clamp amplitude to avoid blowup in the live preview
    float mag = sqrt(na * na + nb * nb);
    if (mag > 4.0) { na *= 4.0 / mag; nb *= 4.0 / mag; }
    f_color = vec4(na, nb, 0.0, 1.0);
}
''')