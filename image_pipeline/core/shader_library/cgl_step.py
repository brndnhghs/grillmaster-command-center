"""cgl_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cgl_step",
          "CGL one Euler step (5-pt Laplacian, toroidal) — complex field in RG",
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
    float alpha = clamp(u_params.x, -3.0, 3.0);   // p1: node alpha (-3..3)
    float beta  = clamp(u_params.y, -3.0, 3.0);   // p2: node beta (-3..3)
    float dt    = clamp(u_params.z, 0.005, 0.2);  // p3: node dt
    float m = a * a + b * b;
    // (1+i*alpha)*lap
    float dispR = lapR - alpha * lapI;
    float dispI = lapI + alpha * lapR;
    // (1+i*beta)*|A|^2*A
    float nlR = m * (a - beta * b);
    float nlI = m * (b + beta * a);
    float na = a + dt * (a + dispR - nlR);
    float nb = b + dt * (b + dispI - nlI);
    // clamp amplitude to avoid blowup in the live preview
    float mag = sqrt(na * na + nb * nb);
    if (mag > 3.0) { na *= 3.0 / mag; nb *= 3.0 / mag; }
    f_color = vec4(na, nb, 0.0, 1.0);
}
''')