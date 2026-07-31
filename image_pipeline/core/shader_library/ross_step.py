"""ross_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ross_step",
          "Rössler array one step: per-cell Rössler ODE + 5-pt diffusive coupling",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float x = s.r, y = s.g, z = s.b;
    // 5-pt Laplacian of each channel
    vec4 ll = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 rr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 dd = texture(u_texture, v_uv + vec2(0.0,-texel.y));
    vec4 uu = texture(u_texture, v_uv + vec2(0.0, texel.y));
    vec3 lap = (ll.rgb + rr.rgb + dd.rgb + uu.rgb - 4.0 * vec3(x, y, z));
    float a = clamp(u_params.x, 0.1, 0.6);
    float b = clamp(u_params.y, 0.05, 0.5);
    float c = clamp(u_params.z, 3.0, 15.0);
    float omega = clamp(u_params.w, 0.5, 2.0);
    float D = 0.5;  // fixed coupling strength (CPU authoritative)
    float dt = 0.08;
    float dx = -omega * y - z + D * lap.x;
    float dy =  omega * x + a * y + D * lap.y;
    float dz =  b + z * (x - c) + D * lap.z;
    vec3 nn = vec3(x, y, z) + dt * vec3(dx, dy, dz);
    nn.z = clamp(nn.z, 0.0, 30.0);  // z always positive in Rössler
    f_color = vec4(nn, 1.0);
}
''')