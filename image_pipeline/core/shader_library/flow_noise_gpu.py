"""flow_noise_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



# ── P0.5 typed-uniform procedural twins: Flow Noise / Spot Noise / Mathematical Marbling ──
# Client-GPU live-preview mirrors of CPU nodes 535 / 534 / 953.
# Every numeric CPU param is bound to a named u_<name> uniform (typed-uniform
# contract). Choice/string params (colormode/palette/source/anim_mode/flow) and the
# CPU-only variable counts (n_spots/n_drops/n_tines) are dropped + justified in
# GPU_PREVIEW_DROP_ALLOW. CPU numpy fns stay authoritative; the GPU twins are
# approximate-by-design live previews (fixed 64-spot / 32-drop / 3-tine loops).
_register("flow_noise_gpu", "Flow Noise — rotating-gradient Perlin (client-GPU twin of node 535)", "procedural", _INFERNO + """
vec3 flow_color(float v){ return inferno(clamp(v,0.0,1.0)); }
void main() {
    vec2 uv = v_uv;
    float t = u_time * u_anim_speed;
    float sc = u_scale * (1.0 + 0.35 * sin(t));
    vec2 p = uv * sc;
    if (u_advect > 0.0) {
        vec2 w = vec2(fbm(p*0.4+3.1), fbm(p*0.4+9.7)) - 0.5;
        p += w * u_advect * 6.0 * (0.5 + 0.5*sin(t*0.7));
    }
    float ang = t * (0.6 + u_spin_var);
    vec2 warp = rot(ang) * vec2(fbm(p*0.5), fbm(p*0.5+21.0));
    float v = 0.0; float amp = 0.5; float norm = 0.0; float ss = 1.0;
    for (int o = 0; o < 6; o++) {
        if (float(o) >= u_octaves) break;
        vec2 q = (p + warp*(1.0+u_spin_var*2.0)) * ss;
        v += amp * fbm(q);
        norm += amp; amp *= 0.5; ss *= 2.0;
    }
    v = (norm > 0.0) ? v/norm : 0.0;
    v = 0.5 + 0.5 * v * u_contrast;
    v = clamp(v, 0.0, 1.0);
    f_color = vec4(flow_color(v), 1.0);
}
""",
    uniforms={
        "scale": {"glsl": "float", "min": 12.0, "max": 260.0, "default": 90.0, "description": "feature size in pixels (lattice spacing)"},
        "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 4.0, "description": "fractal octaves (turbulent detail)"},
        "spin_var": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "0 = uniform global spin, 1 = per-cell chaotic spin"},
        "advect": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.2, "description": "pseudo-advection strength (domain transport)"},
        "contrast": {"glsl": "float", "min": 0.4, "max": 2.5, "default": 1.15, "description": "final tone contrast"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed multiplier"},
    }
    )