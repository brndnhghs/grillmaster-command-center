"""grayscott_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("grayscott_step",
          "Gray-Scott one Euler step (5-point Laplacian, toroidal) — reads/writes RG state",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float U = s.r, V = s.g;
    // 5-point Laplacian; RepeatWrapping on the state texture makes it toroidal,
    // matching the CPU np.roll boundaries.
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapU = sl.r + sr.r + su.r + sd.r - 4.0 * U;
    float lapV = sl.g + sr.g + su.g + sd.g - 4.0 * V;
    float F  = u_params.x;   // feed
    float k  = u_params.y;   // kill
    float Du = u_params.z;   // diffusion U
    float Dv = u_params.w;   // diffusion V
    float uvv = U * V * V;
    const float dt = 1.0;    // CPU default timestep; substeps control pace
    float nU = U + dt * (Du * lapU - uvv + F * (1.0 - U));
    float nV = V + dt * (Dv * lapV + uvv - (F + k) * V);
    f_color = vec4(clamp(nU, 0.0, 1.0), clamp(nV, 0.0, 1.0), 0.0, 1.0);
}
''')