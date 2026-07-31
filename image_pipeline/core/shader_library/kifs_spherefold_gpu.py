"""kifs_spherefold_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



_register("kifs_spherefold_gpu", "Kaleidoscopic IFS - box-fold + sphere-fold (Knighty/Kali 2010)", "procedural",
          _FRACTAL_HELPERS + '''
// KIFS (Kaleidoscopic Iterated Function System) fractal on the 2D plane.
// Core technique (Knighty / Kali, 2010 "Kaleidoscopic IFS" thread):
//   1) box fold  - mirror z into the positive wedge + diagonal fold
//   2) scale + offset  - the "kaleidoscope" growth
//   3) rotation between folds
//   4) sphere fold - pull near-origin points outward, push far points in
//      (the minR/maxR radius clamp that gives the characteristic holes).
// Iterating these affine folds produces the iconic self-similar kaleidoscope.
// Distinct from kifs_gpu (node 402), which only does a wedge + scale fold.
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 2.2;

    // Global spin for the live preview (t=0 -> identity, so a static frame
    // still renders the canonical fractal).
    float ta = u_time * 0.2;
    uv = rot(ta) * uv;

    float scale = u_scale;                       // negative scale = classic detail
    float ang = (u_fold_angle - 0.5) * 6.2831853 + u_time * 0.13;  // per-iteration rot
    float minR = max(u_min_r, 0.05);
    float maxR = max(u_max_r, minR + 0.05);
    float minR2 = minR * minR;
    float maxR2 = maxR * maxR;
    // Time-orbiting offset -> the whole kaleidoscope breathes and flows.
    vec2 offs = vec2(1.0 + 0.30 * sin(u_time * 0.7),
                     0.5 + 0.30 * cos(u_time * 0.9));
    float color_shift = u_color_shift;

    vec2 z = uv;
    float trap = 1e9;
    float esc = 0.0;
    float escR2 = 64.0;

    for (int i = 0; i < 16; i++) {
        // 1) box fold - mirror into the first octant, then diagonal fold
        z = abs(z);
        if (z.x < z.y) z = z.yx;
        // 2) scale + offset
        z = z * scale + offs;
        // 3) rotation between folds
        z = rot(ang) * z;
        // 4) sphere fold (radius clamp)
        float r2 = dot(z, z);
        if (r2 < minR2) {
            z *= minR2 / max(r2, 1e-6);
        } else if (r2 > maxR2) {
            z *= maxR2 / r2;
        }
        trap = min(trap, length(z));
        if (dot(z, z) > escR2) { esc = 1.0; break; }
    }

    float v;
    if (esc > 0.5) {
        v = clamp(trap / (maxR * 2.0), 0.0, 1.0);
    } else {
        v = 0.0;  // interior -> dark
    }
    v = clamp(v + color_shift, 0.0, 1.0);
    vec3 col = fractal_palette(v);
    col = mix(vec3(0.02, 0.02, 0.06), col, esc);  // interior fill
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "scale": {"glsl": "float", "min": -3.0, "max": 1.0, "default": -1.8, "description": "fold scale (negative = classic kaleidoscopic detail)"},
    "fold_angle": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "per-iteration rotation slot (0.5 -> 0 rad)"},
    "min_r": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.5, "description": "sphere-fold inner radius"},
    "max_r": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 2.0, "description": "sphere-fold outer radius"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"}
}
    )