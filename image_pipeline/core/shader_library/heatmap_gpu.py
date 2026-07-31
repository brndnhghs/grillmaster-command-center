"""heatmap_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _inferno_local



# ── Node 322: Procedural Phasor Noise (closed-form f(uv,t)) ──
# Research technique: "Procedural Phasor Noise", Tricard, Efremov, Zanni, Neyret,
# Martínez, Lefebvre — ACM TOG / SIGGRAPH 2019.
# Reference: http://thibaulttricard.fr/project_page/phasor_noise/phasor.html
# Phasor noise reformulates Gabor noise as a complex PHASOR field: a sum of
# complex-valued Gabor kernels g_i(x) = A_i·exp(-π b²|x-x_i|²)·exp(i·2π f·(x-x_i)).
# Summing kernels accumulates real+imag parts; the ARGUMENT (phase) of the sum is
# the phasor field. Taking sin(phase) yields oscillating ridge patterns whose
# CONTRAST is decoupled from local intensity (unlike raw Gabor noise), giving the
# characteristic fingerprint/wood-grain ridges with locally controllable
# frequency and orientation. Closed-form per pixel → exact GPU live preview;
# CPU numpy stays authoritative for export.

_register("heatmap_gpu",
          "Density heatmap (client-GPU twin of node 43)",
          "procedural", _inferno_local('') + '''
void main() {
    // u_params.x = sigma proxy (0.5 -> ~0.06), u_params.z = colormap_shift.
    float sigma = 0.01 + u_sigma * 0.10;
    float shift = u_colormap_shift;
    float t = u_time * 0.04;

    // Closed-form kernel-density estimate from K drifting gaussian clusters.
    vec2 p = v_uv;
    float dens = 0.0;
    const int K = 18;
    for (int i = 0; i < K; i++) {
        float fi = float(i);
        vec2 c = vec2(0.15 + 0.7 * hash21(vec2(fi, 3.3)),
                      0.15 + 0.7 * hash21(vec2(fi, 7.7)));
        c += 0.04 * vec2(sin(t + fi), cos(t * 1.1 + fi * 1.7));
        float d2 = dot(p - c, p - c);
        dens += exp(-d2 / (2.0 * sigma * sigma + 1e-4));
    }
    dens = clamp(dens * 0.18, 0.0, 1.0);
    dens = fract(dens + shift);
    f_color = vec4(inferno(dens), 1.0);
}
''',
    uniforms={
    "sigma": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "gaussian sigma"},
    "colormap_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "colormap shift"}
}
    )