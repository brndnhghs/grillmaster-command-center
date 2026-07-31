"""oscillon_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 166: Parametric Oscillator Lattice (Oscillon Resonance) ── ------------
# d2u/dt2 = D*lap - gamma*v - w0^2(1 + eps*sin p)*u - beta*u^3
# p1=epsilon, p2=omega0, p3=damping gamma, p4=diffusion D. Beta fixed 0.3.
_register("oscillon_seed",
          "Oscillon Resonance seed: multi-scale hashed noise displacement (node 166 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 5.0) * 0.5 + noise(v_uv * 13.0) * 0.3
            + noise(v_uv * 28.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.4, 0.0, 0.0, 1.0);  // R=u (small), G=v, B=phase
}
''')