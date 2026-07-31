"""dither13_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# 74 Swirl Displacement — polar swirl remap (GPU live twin)

# 13 Dithering — Bayer 8x8 ordered dither with N-level quantization (GPU live
# twin). The CPU node's default `fs` (Floyd-Steinberg error diffusion) is an
# inherently serial scan that cannot be reproduced per-pixel on the GPU, so this
# twin renders the ORDERED (Bayer) approximation and the CPU fn stays
# authoritative for all error-diffusion algorithms. `levels` -> p1 (2..8),
# `contrast` -> p2. `algorithm`/`palette`/`noise_type` are string choices
# (pitfall #14) and are left unmapped.
_register("dither13_gpu", "GPU Bayer-4 ordered dithering (node 13 twin)", "filter", _filter_typed('''
    float gray = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float contrast = 0.5 + u_contrast * 2.5;            // contrast
    gray = clamp((gray - 0.5) * contrast + 0.5, 0.0, 1.0);
    // Bayer 4x4 ordered threshold matrix (values 0..15) via a small lookup.
    float bx = mod(gl_FragCoord.x, 4.0);
    float by = mod(gl_FragCoord.y, 4.0);
    float m[16];
    m[0]=0.0;  m[1]=8.0;  m[2]=2.0;  m[3]=10.0;
    m[4]=12.0; m[5]=4.0;  m[6]=14.0; m[7]=6.0;
    m[8]=3.0;  m[9]=11.0; m[10]=1.0; m[11]=9.0;
    m[12]=15.0;m[13]=7.0; m[14]=13.0;m[15]=5.0;
    int idx = int(by) * 4 + int(bx);
    float threshold = (m[idx] + 0.5) / 16.0;            // in (0,1)
    float levels = floor(2.0 + u_levels * 6.0 + 0.5); // levels
    float steps = max(1.0, levels - 1.0);
    float scaled = gray * steps;
    float lower = floor(scaled);
    float frac = scaled - lower;
    float q = (lower + (frac > threshold ? 1.0 : 0.0)) / steps;
    f_color = vec4(vec3(clamp(q, 0.0, 1.0)), 1.0);
'''), uniforms={
        "levels": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "quantization levels"},
        "contrast": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "contrast"},
    })