"""sine_gordon_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sine_gordon_step",
          "Sine-Gordon one step (leapfrog): v += c2*lap - G*sin(u); u += v + drive",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float u = s.r, v = s.g;
    float lu = texture(u_texture, v_uv + vec2(-texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2( texel.x, 0.0)).r
             + texture(u_texture, v_uv + vec2(0.0, texel.y)).r
             + texture(u_texture, v_uv + vec2(0.0,-texel.y)).r - 4.0 * u;
    float c = clamp(u_params.x, 0.5, 1.5);
    float c2 = min(0.20 * c * c, 0.45);
    float damp = clamp(u_params.y, 0.95, 1.0);
    float G = clamp(u_params.z, 0.1, 4.0);
    float S = 0.20 * G;
    float A = clamp(u_params.w, 0.0, 2.0);
    float drive = A * 0.05 * (sin(6.2831853 * 3.0 * v_uv.x) + sin(6.2831853 * 3.0 * v_uv.y));
    float vn = (v + c2 * lu - S * sin(u)) * damp;
    float un = u + vn + drive;
    f_color = vec4(clamp(un, -8.0, 8.0), clamp(vn, -8.0, 8.0), 0.0, 1.0);
}
''')