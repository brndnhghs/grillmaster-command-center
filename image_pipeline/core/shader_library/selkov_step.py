"""selkov_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("selkov_step",
          "Sel'kov one Euler step (5-pt Laplacian, toroidal) — excitable u²v kinetics",
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
    float a  = u_params.x;   // substrate supply
    float b  = u_params.y;   // intermediate removal
    float Du = u_params.z;   // diffusion U
    float Dv = u_params.w;   // diffusion V
    float uvv = U * U * V;
    float dt = 0.2;          // matches CPU default; substeps control pace
    float nU = U + dt * (a - U + uvv + Du * lapU);
    float nV = V + dt * (b * U * U - uvv + Dv * lapV);
    f_color = vec4(clamp(nU, 0.0, 2.0), clamp(nV, 0.0, 2.0), 0.0, 1.0);
}
''')