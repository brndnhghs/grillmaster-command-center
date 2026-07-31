"""kifs_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



# ── Kaleidoscopic IFS (client-GPU twin of node 402) ──
# KIFS: repeated mirror-fold + n-fold wedge fold + rotation + scale + offset.
# The mirror+wedge fold is what yields the snowflake/kaleidoscopic symmetry and
# the negative scale yields the classic self-similar detail. This is a
# closed-form function of (uv, t), so the twin is an exact parity preview; the
# server's CPU numpy node (method_kaleidoscopic_ifs) stays authoritative.
# Neutral: 0.5 on every slot yields the node's canonical default view
# (sym=6, scale=-2, no animation). p1=scale, p2=fold_angle, p3=symmetry(2..8),
# p4=color_shift. anim_mode/offset/colormode are choice strings (pitfall #14)
# so they are NOT mapped — the live preview shows the node's default orbit
# coloring at the default offsets.
_register("kifs_gpu", "Kaleidoscopic IFS fractal (client-GPU twin of node 402)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 3.0;  // match CPU: map pixels to a ~[-3,3] complex-plane region

    // u_time drives a live-preview spin (matches the CPU 'spin' anim mode at
    // t=0; the client advances u_time so the live preview animates). At t=0
    // this is the identity rotation, so parity with the static CPU frame holds.
    float ta = u_time * 0.3;
    uv = rot(ta) * uv;

    float scale = u_scale;                       // 0.5 -> -2.0 canonical
    float ang = (u_fold_angle - 0.5) * 6.2831853; // 0.5 -> 1.0 rad canonical
    float sym = floor(2.0 + u_symmetry * 6.0);    // 0.5 -> 6-fold canonical
    float color_shift = u_color_shift;
    vec2 offs = vec2(1.0, 1.0);                   // default offset (unmapped)

    float r2esc = 144.0;                          // escape_radius = 12 -> r2=144
    float trap = 1e9;
    float esc = 0.0;

    for (int i = 0; i < 24; i++) {
        // 1) Mirror + n-fold wedge fold (kaleidoscopic symmetry)
        vec2 z = abs(uv);
        float a = atan(z.y, z.x);
        float r = length(z);
        float wedge = 3.14159265 / sym;
        a = mod(a, 2.0 * wedge);
        if (a > wedge) a = 2.0 * wedge - a;
        z = vec2(cos(a), sin(a)) * r;
        // 2) Rotation between folds
        z = rot(ang) * z;
        // 3) Scale + offset
        z = z * scale + offs;
        uv = z;
        trap = min(trap, length(z));
        if (dot(z, z) > r2esc) { esc = 1.0; break; }
    }

    float v;
    if (esc > 0.5) {
        v = trap / 12.0;
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
    "scale": {"glsl": "float", "min": -3.0, "max": 1.0, "default": -2.0, "description": "fold scale (negative = classic detail)"},
    "fold_angle": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "fold rotation slot (0.5 -> 1.0 rad)"},
    "symmetry": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "kaleidoscopic symmetry order (0.5 -> 6)"},
    "color_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "palette color offset"}
}
    )