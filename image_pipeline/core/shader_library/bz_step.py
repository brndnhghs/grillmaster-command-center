"""bz_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("bz_step",
          "BZ Oregonator one step (5-pt Laplacian, toroidal) — Oregonator kinetics",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + su.r + sd.r - 4.0 * U;
    float lapV = sl.g + sr.g + su.g + sd.g - 4.0 * V;
    float eps = u_params.x;   // epsilon (timescale separation)
    float q   = u_params.y;   // q
    float f   = u_params.z;   // f
    float Du  = u_params.w;   // diffusion U (Dv ~ 0 for classic BZ)
    float uvq = (U + q) > 0.0 ? (U * V * (U - q) / (U + q)) : 0.0;
    float dU = (U - U * U - f * uvq + Du * lapU) / max(eps, 1e-3);
    float dV = U - V + 0.0 * lapV;   // Dv ~ 0 -> V is reaction-dominated
    float nU = U + dU * 0.02;
    float nV = V + dV * 0.02;
    f_color = vec4(clamp(nU, 0.0, 1.0), clamp(nV, 0.0, 1.0), 0.0, 1.0);
}
''')