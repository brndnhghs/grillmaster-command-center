"""anisotropic_kuwahara_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# Anisotropic Kuwahara — coherence-enhancing painterly abstraction (node 68 twin).
# A rotated, elongated Gaussian kernel is oriented along the local structure-tensor
# direction; the window is split into four angular sectors and the lowest-variance
# sector wins (Kyprianidis et al. 2011). Decoded from normalized u_params:
# p1 = radius∈[0,1] (0.5→8), p2 = anisotropy∈[0,1] (0.5→4). CPU numpy node stays
# authoritative for exact export.
_register("anisotropic_kuwahara_gpu", "GPU anisotropic Kuwahara painterly abstraction", "filter", _filter_typed('''
    // Decode normalized params (client sends radius/anisotropy 0..1 per the
    // 0.5-neutral GPU contract). radius 0.5→8 (node default), aniso 0.5→4.
    int R = int(clamp(2.0 + u_radius * 12.0, 2.0, 15.0));
    float aniso = clamp(1.0 + u_anisotropy * 6.0, 1.0, 12.0);

    // Local structure via Sobel (step is reserved in filter twins — only read it).
    float gx = 0.0, gy = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float v = dot(texture(u_texture, uv + vec2(float(x), float(y)) * step).rgb,
                          vec3(0.299, 0.587, 0.114));
            gx += float(x) * v;
            gy += float(y) * v;
        }
    }
    float theta = 0.5 * atan(gy, gx + 1e-5);
    float c = cos(-theta), s = sin(-theta);
    float sx = float(R) * 0.6 * sqrt(aniso);
    float sy = float(R) * 0.6 / sqrt(aniso);

    vec3 sc[4]; vec3 sc2[4]; float sw[4];
    for (int q = 0; q < 4; q++) { sc[q] = vec3(0.0); sc2[q] = vec3(0.0); sw[q] = 0.0; }

    for (int i = -8; i <= 8; i++) {
        for (int j = -8; j <= 8; j++) {
            float xe = float(i) * c - float(j) * s;
            float ye = float(i) * s + float(j) * c;
            float w = exp(-(xe * xe / (2.0 * sx * sx) + ye * ye / (2.0 * sy * sy)));
            int quad = (xe > 0.0 ? 2 : 0) + (ye > 0.0 ? 1 : 0);
            vec3 col = texture(u_texture, uv + vec2(float(i), float(j)) * step).rgb;
            sc[quad] += col * w;
            sc2[quad] += col * col * w;
            sw[quad] += w;
        }
    }

    float bestVar = 1e9;
    vec3 bestC = vec3(0.0);
    for (int q = 0; q < 4; q++) {
        if (sw[q] > 1e-4) {
            vec3 m = sc[q] / sw[q];
            vec3 m2 = sc2[q] / sw[q];
            float var = dot(m2 - m * m, vec3(1.0));
            if (var < bestVar) { bestVar = var; bestC = m; }
        }
    }
    f_color = vec4(bestC, 1.0);
'''), uniforms={
        "radius": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "Kuwahara radius"},
        "anisotropy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "anisotropy"},
    })