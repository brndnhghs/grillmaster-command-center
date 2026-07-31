"""faraday_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 144: Faraday Waves ── -------------------------------------------------
# Parametrically-driven damped wave: force = nu*lap - gamma*v - (w0^2 + A*cos p)*u + a*u^3
# p1=amplitude A, p2=omega0, p3=damping gamma, p4=capillary nu. Alpha fixed 0.5.
_register("faraday_seed",
          "Faraday Waves seed: multi-scale hashed noise height field (node 144 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 6.0) * 0.5 + noise(v_uv * 14.0) * 0.3
            + noise(v_uv * 30.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.5, 0.0, 0.0, 1.0);  // R=h (small), G=v, B=phase
}
''')