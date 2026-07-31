"""ifs_fractal_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── IFS Fractal attractor (node 353 twin) ──────────────────────────────────────
# Per-pixel chaos game: orbit start seeded by pixel+time, 30-50 iterations
# through a preset's affine maps, final position deposited as soft density.
# Named typed uniforms mirror the CPU node's numeric params; preset/coloring/
# anim_mode are choice strings (pitfall #14) dropped for GPU preview.
_register("ifs_fractal_gpu",
          "IFS Fractal attractor — chaos game (client-GPU twin of node 353)",
          "procedural", '''vec3 ifspal(float t, float shift) {
    t = clamp(t + shift, 0.0, 1.0);
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec2 hash22in(vec2 p) {
    float a = hash21(p), b = hash21(p + vec2(1.0, 0.0));
    return vec2(a, b);
}

vec2 ifs_map(vec2 z, int preset, int ch) {
    float h3 = 0.3333333, h6 = 0.1666667;
    if (preset == 0) { // sierpinski
        if (ch == 0) return vec2(h3*z.x, h3*z.y);
        else if (ch == 1) return vec2(h3*z.x + h3, h3*z.y);
        else return vec2(h3*z.x + h6, h3*z.y + 0.4330127);
    } else if (preset == 1) { // dragon
        float s = 0.353553;
        if (ch == 0) return vec2(s*(z.x - z.y), s*(z.x + z.y));
        else return vec2(-s*(z.x + z.y) + 1.0, s*(z.x - z.y));
    } else { // snowflake
        float a = float(ch) * 1.0471976;
        float c = cos(a), sn = sin(a);
        return vec2(h3*(c*z.x - sn*z.y) + 0.5, h3*(sn*z.x + c*z.y) + 0.5);
    }
}

void main() {
    vec2 uv = v_uv;
    int preset = int(u_preset + 0.5);
    float density = 0.0;
    int N = int(u_points * 0.0004 + 8.0);
    if (N > 60) N = 60;
    float t_rot = u_time * 0.25;
    float ca = cos(t_rot), sa = sin(t_rot);

    for (int orbit = 0; orbit < 8; orbit++) {
        vec2 z = hash22in(vec2(float(orbit)*1.7+0.5, float(orbit)*3.1+0.3));
        z = z * 2.0 - 1.0;
        int nmaps = (preset == 1) ? 2 : 6;
        for (int i = 0; i < 40; i++) {
            float h = fract(sin(dot(z + float(i)*0.17, vec2(127.1, 311.7))) * 43758.5453);
            z = ifs_map(z, preset, int(h * float(nmaps)));
        }
        float ox = z.x - 0.5, oy = z.y - 0.5;
        vec2 sp = vec2(ca*ox - sa*oy, sa*ox + ca*oy) + 0.5;
        density += exp(-length(uv - sp) * 22.0);
    }
    density = clamp(density / 8.0, 0.0, 1.0);
    f_color = vec4(ifspal(density * 1.3, u_hue_shift), 1.0);
}
''', uniforms={
    "preset": {"glsl": "choice", "choices": ["sierpinski", "dragon", "snowflake"], "default": "sierpinski", "description": "IFS preset family"},
    "points": {"glsl": "float", "min": 10000.0, "max": 500000.0, "default": 120000.0, "description": "chaos-game iterations (density proxy)"},
    "hue_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "hue rotation"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed multiplier"},
})