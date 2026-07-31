"""sine_gordon_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 499: Sine-Gordon Equation ── ------------------------------------------
# 2D Sine-Gordon u_tt = c^2 lap(u) - G*sin(u) + A*drive. Same leapfrog (u, v)
# fields as the Wave Equation (node 100) with the addition of the -G*sin(u)
# restoring term that produces kink/antikink solitons and breathers.
# p1=wave_speed (c), p2=damping, p3=coupling (G), p4=drive_amplitude (A).
# c2 = min(0.20*c*c, 0.45) (CFL-safe); S = 0.20*G; drive = A*0.05*(sin6.28*3x+sin6.28*3y).
_register("sine_gordon_seed",
          "Sine-Gordon seed: kink-antikink initial displacement, zero velocity (node 499 twin)",
          "procedural", '''
void main() {
    float k = 8.0;
    float x = v_uv.x;
    float u0 = 4.0 * (atan(exp(k * (x - 0.35))) - atan(exp(k * (x - 0.65))));
    f_color = vec4(u0, 0.0, 0.0, 1.0);  // R=u (kink pair), G=v=0
}
''')