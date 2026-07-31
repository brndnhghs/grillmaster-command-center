"""gyroid_tpms_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO




# -- Categorical coverage: Gyroid TPMS (typed GPU node 360) --
# Closed-form triply-periodic minimal-surface field (Schoen 1970 gyroid,
# Schwarz 1890 P/D, Neovius, Schoen I-WP) evaluated on a swept slice plane.
# Mirrors CPU node 964 (patterns/gyroid_tpms.py); the third coordinate z is
# animated with u_time so the 2D cross-section morphs continuously as the plane
# sweeps through the 3D volume -> genuinely time-varying, survives the
# contrast-only static liveness cull. CPU numpy node 964 stays authoritative
# for export (two-tier precision). Inferno colormap for the default view.
_register("gyroid_tpms_gpu",
          "Gyroid TPMS -- closed-form triply-periodic minimal-surface shell "
          "on a swept slice plane (typed GPU twin of node 964); genuinely "
          "animated (slice z advances with u_time)",
          "procedural", _INFERNO + '''
void main() {
    vec2 p = (v_uv - 0.5) * 2.0 * 3.14159265 * max(u_freq, 1e-3);
    float t = u_time * 0.6;
    // Optional lattice rotation.
    float rot_ang = u_rotate * t * 0.5;
    float ca = cos(rot_ang), sa = sin(rot_ang);
    p = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
    // Domain warp (animated so warp>0 stays alive).
    float wstr = u_warp;
    if (wstr > 0.0) {
        p.x += wstr * sin(p.y * 0.5 + t);
        p.y += wstr * cos(p.x * 0.5 + t * 0.8);
    }
    float z = t;   // swept slice plane
    float sx = sin(p.x), cx = cos(p.x);
    float sy = sin(p.y), cy = cos(p.y);
    float sz = sin(z),   cz = cos(z);

    int surf = int(clamp(u_surface + 0.5, 0.0, 4.0));
    float g, gmax;
    if (surf == 1) {            // schwarz_p
        g = cx + cy + cz; gmax = 3.0;
    } else if (surf == 2) {     // diamond (Schwarz D)
        g = sx*sy*sz + sx*cy*cz + cx*sy*cz + cx*cy*sz; gmax = 1.5;
    } else if (surf == 3) {     // neovius
        g = 3.0*(cx+cy+cz) + 4.0*cx*cy*cz; gmax = 3.0;
    } else if (surf == 4) {     // I-WP (Schoen)
        g = 2.0*(cx*cy + cy*cz + cz*cx) - (cos(2.0*p.x)+cos(2.0*p.y)+cos(2.0*z));
        gmax = 3.0;
    } else {                    // gyroid (Schoen)
        g = sx*cy + sy*cz + sz*cx; gmax = 1.5;
    }
    float gn = g / gmax;        // ~[-1,1]
    float lvl = u_level / gmax;

    float val;
    if (u_shell > 0.5) {
        float d = abs(gn - lvl);
        val = clamp(1.0 - d / max(u_thickness, 1e-6), 0.0, 1.0);
    } else {
        val = clamp((gn - lvl) * 0.5 + 0.5, 0.0, 1.0);
    }
    val = clamp(0.5 + (val - 0.5) * u_contrast, 0.0, 1.0);
    vec3 col = inferno(val);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "surface":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 0.0, "description": "minimal surface: 0 gyroid, 1 schwarz_p, 2 diamond, 3 neovius, 4 iwp"},
    "freq":      {"glsl": "float", "min": 1.0, "max": 16.0, "default": 5.0, "description": "spatial frequency (cells across canvas)"},
    "level":     {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.0, "description": "iso-level of the surface (shifts shell in/out)"},
    "thickness": {"glsl": "float", "min": 0.02, "max": 0.8, "default": 0.22, "description": "shell half-thickness of the surface band"},
    "warp":      {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.0, "description": "domain-warp strength (organic distortion)"},
    "contrast":  {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.2, "description": "final tone contrast"},
    "shell":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "render mode: 0 smooth field, 1 shell band"},
    "rotate":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "lattice rotation amount (0 off, 1 spin with time)"},
})