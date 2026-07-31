"""symmetric_icon_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Symmetric Icon attractor (node 416 twin) ───────────────────────────────────
# Orbits the Field & Golubitsky symmetric-chaos map per pixel, deposits density.
# Named typed uniforms mirror the CPU node's numeric params; colormode/palette/
# source/anim_mode are choice strings (pitfall #14) dropped for preview.
_register("symmetric_icon_gpu",
          "Symmetric Icon attractor — Field & Golubitsky (client-GPU twin of node 416)",
          "procedural", '''vec3 iconpal(float t, float shift) {
    t = clamp(t + shift, 0.0, 1.0);
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec2 cpow(vec2 z, int n) {
    vec2 r = vec2(1.0, 0.0);
    for (int i = 0; i < 10; i++) {
        if (i >= n) break;
        r = vec2(r.x*z.x - r.y*z.y, r.x*z.y + r.y*z.x);
    }
    return r;
}

vec2 hash22in(vec2 p) {
    float a = hash21(p), b = hash21(p + vec2(1.0, 0.0));
    return vec2(a, b);
}

void main() {
    int sym = int(u_symmetry + 0.5);
    vec2 uv = v_uv;
    float density = 0.0;
    float t_rot = u_time * 0.3;
    float time_warp = 1.0 + 0.15 * sin(u_time * 0.7);

    for (int orbit = 0; orbit < 8; orbit++) {
        vec2 z = hash22in(vec2(float(orbit)*1.3+0.5, float(orbit)*2.9+0.3));
        z = z * 1.4 - 0.7;
        for (int i = 0; i < 32; i++) {
            float z2 = dot(z, z);
            vec2 zn = cpow(z, sym);
            vec2 coeff = vec2(u_a0 + u_a1*z2 + u_a2*zn.x, u_a3);
            vec2 conj_pow = cpow(vec2(z.x, -z.y), sym - 1);
            z = coeff * z + u_a4 * conj_pow;
        }
        // subtle time-warp of position (anim_mode=evolve analogue)
        z = vec2(z.x + 0.003 * sin(u_time * 1.1 + float(orbit)), z.y);
        float ox = z.x * 0.7 + 0.5;
        float oy = z.y * 0.7 + 0.5;
        float c = cos(t_rot), s = sin(t_rot);
        vec2 sp = vec2(c*(ox-0.5) - s*(oy-0.5) + 0.5, s*(ox-0.5) + c*(oy-0.5) + 0.5);
        density += exp(-length(uv - sp) * 20.0);
    }
    density = clamp(density / 8.0, 0.0, 1.0);
    float v = density * time_warp;
    f_color = vec4(iconpal(v * 1.1, u_palette_shift), 1.0);
}
''', uniforms={
    "symmetry": {"glsl": "int", "min": 2, "max": 9, "default": 6, "description": "rotational symmetry order n"},
    "a0": {"glsl": "float", "min": -3.0, "max": 3.0, "default": -2.0, "description": "real constant term"},
    "a1": {"glsl": "float", "min": -3.0, "max": 3.0, "default": 1.5, "description": "modulus |z|^2 coupling"},
    "a2": {"glsl": "float", "min": -3.0, "max": 3.0, "default": -0.1, "description": "Re(z^n) coupling"},
    "a3": {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.0, "description": "imaginary term (chiral twist)"},
    "a4": {"glsl": "float", "min": -1.5, "max": 1.5, "default": 0.6, "description": "conjugate z^(n-1) coupling"},
    "palette_shift": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "cosine palette hue offset"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed multiplier"},
    "seed_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "blend with wired luminance"},
})