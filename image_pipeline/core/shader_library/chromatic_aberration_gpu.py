"""chromatic_aberration_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed-uniform shims for CPU filter nodes 417 / 419 ──
# Mirrors node 417 (Chromatic Aberration) and node 419 (Thin-Film Interference)
# with NAMED typed uniforms that equal the CPU node's real params (contract #5),
# so the live preview tracks the sliders. The CPU numpy node stays authoritative
# for export (two-tier precision). Each uniform is verified live by
# test_typed_uniforms_drive_output (MAD >= 1.0 when perturbed to an extreme).
_register("chromatic_aberration_gpu", "Chromatic aberration RGB split (client-GPU twin of node 417)",
          "filter", '''
void main() {
    // Optical center can be nudged by center_drift (kept static here — the CPU
    // node only orbits it in spin mode); at the default 0.4 it sits at (0.5,0.5).
    vec2 ctr = vec2(0.5) + (u_center_drift - 0.4) * vec2(0.25, -0.15);
    vec2 d = v_uv - ctr;
    float rn = length(d);
    vec2 dir = d / max(rn, 1e-4);
    // Lateral split grows as r^curve (k=2 reproduces physical lateral CA).
    float amt = u_amount * 0.012;
    float k = amt * pow(rn, u_curve);
    // Optional barrel/pincushion radial distortion.
    float rbar = rn * (1.0 + u_barrel * rn * rn);
    float rR = rbar + k;          // R sampled outward
    float rB = rbar - k;          // B sampled inward
    float rC = texture(u_texture, ctr + dir * rR).r;
    float gC = texture(u_texture, v_uv).g;
    float bC = texture(u_texture, ctr + dir * rB).b;
    vec3 col = vec3(rC, gC, bC);
    col *= (1.0 - u_vignette * rn * rn);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "amount":       {"glsl": "float", "min": 0.0, "max": 60.0, "default": 20.0,
                    "description": "max lateral RGB split (px)"},
    "curve":        {"glsl": "float", "min": 1.0, "max": 4.0, "default": 2.0,
                    "description": "radial falloff exponent"},
    "barrel":       {"glsl": "float", "min": -0.4, "max": 0.4, "default": 0.0,
                    "description": "barrel/pincushion distortion"},
    "vignette":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                    "description": "edge darkening"},
    "center_drift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4,
                    "description": "aberration-center offset"},
})