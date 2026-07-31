"""gpe_step — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("gpe_step",
          "GPE one symplectic Euler step (5-pt Laplacian proxy for kinetic) — complex field in RG",
          "procedural", '''
void main() {
    // NOTE: u_time is 0 here (renderGpuSim passes no time to step shaders,
    // pitfall #6b). We carry an accumulating sim-time in the .b state channel
    // so the stirrer orbits and the live preview actually moves frame to frame.
    vec2 texel = 1.0 / u_resolution;
    vec4 s = texture(u_texture, v_uv);
    float a = s.r, b = s.g;
    float tt = s.b;                            // accumulated sim-time (advances each step)
    vec4 sl = texture(u_texture, v_uv + vec2(-texel.x, 0.0));
    vec4 sr = texture(u_texture, v_uv + vec2( texel.x, 0.0));
    vec4 su = texture(u_texture, v_uv + vec2(0.0,  texel.y));
    vec4 sd = texture(u_texture, v_uv + vec2(0.0, -texel.y));
    float lapR = sl.r + sr.r + su.r + sd.r - 4.0 * a;
    float lapI = sl.g + sr.g + su.g + sd.g - 4.0 * b;
    float gnl  = clamp(u_params.x, 0.0, 4.0);    // p1: node nonlinearity g (0.2..4.0)
    float ss   = clamp(u_params.y, 0.02, 1.5);   // p2: node stir_speed (0.05..1.5)
    float alpha = clamp(u_params.z, 0.02, 2.0);  // p3: node alpha / kinetic coeff (0.05..2.0)
    float stp  = clamp(u_params.w, 0.0, 20.0);   // p4: node stir_amp (1..20)
    float dt = 0.05;                             // fixed live-preview timestep
    // Moving repulsive stirrer (single gaussian, orbits via accumulated time)
    vec2 ctr = vec2(0.5 + 0.18 * sin(tt * 0.6 * ss), 0.5 + 0.14 * cos(tt * 0.5 * ss));
    vec2 dd = v_uv - ctr;
    float V = stp * 6.0 * exp(-dot(dd, dd) * 30.0);
    float m = a * a + b * b;
    float D = 0.5 * (gnl * m + V) * dt;          // half-step potential phase
    float c = cos(D), sn = sin(D);
    float a1 = a * c - b * sn;
    float b1 = b * c + a * sn;
    // kinetic step (5-pt Laplacian proxy for spectral k²)
    float a2 = a1 + alpha * lapI * dt;
    float b2 = b1 - alpha * lapR * dt;
    float mag = sqrt(a2 * a2 + b2 * b2);
    if (mag > 4.0) { a2 *= 4.0 / mag; b2 *= 4.0 / mag; }
    f_color = vec4(a2, b2, tt + dt, 1.0);        // .b carries the advancing sim-time
}
''')