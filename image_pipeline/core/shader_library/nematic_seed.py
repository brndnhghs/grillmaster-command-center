"""nematic_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.6 — Active Nematic Liquid Crystals (node 99) Q-tensor field twin
# ═══════════════════════════════════════════════════════════════════════════
# Simplified Landau-de Gennes Q-tensor model. State packs the two independent
# components of the traceless symmetric 2×2 tensor: Qxx in .r, Qxy in .g. Explicit
# Euler: ∂Q/∂t = Γ·H + α·Q + D·∇²Q + noise, with H = -(A·Q + C·Tr(Q²)·Q). Thermal
# noise (which nucleates the ±½ defects) is reproduced as state-dependent hash
# noise that varies each substep. Display = schlieren texture (director hue,
# order-parameter brightness) + defect glow. CPU numpy node authoritative.
_register("nematic_seed",
          "Active-nematic seed: small random Q-tensor (Qxx=.r, Qxy=.g) (node 99)",
          "procedural", '''
void main() {
    float a = hash21(v_uv * 311.0 + 1.7) - 0.5;
    float b = hash21(v_uv * 517.0 + 9.3) - 0.5;
    f_color = vec4(a * 0.12, b * 0.12, 0.0, 1.0);   // ~N(0,0.05) initial order
}
''')