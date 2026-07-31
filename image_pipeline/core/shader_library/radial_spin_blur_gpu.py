"""radial_spin_blur_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ══════════════════════════════════════════════════════════════════════════
#  CPU-node closed-form filter twins (Route 0 / GPU-First gap mirror)
#  Live-preview GPU twins for filter nodes whose CPU algorithm is a faithful
#  per-pixel f(uv, input) operator. CPU node stays authoritative for export;
#  these drive the client-side live preview. Choice params (blur_type /
#  source / tint / palette / anim_mode / combine / output / n_orientations)
#  are intentionally omitted from param_map — the twins animate continuously
#  from u_time so the preview is always live, and the CPU export honours the
#  exact choices. Helper functions are the prologue's (rot / hash21 / fbm);
#  `step` is the prologue-reserved vec2 so the bodies use `mix`/manual compares
#  (never the `step()` builtin). Animation uses cos(u_time)/linear terms so the
#  t=0 vs t=π audit is never a sin-phase false negative.
# ══════════════════════════════════════════════════════════════════════════

# ── 486 Radial & Spin Blur (client-GPU twin) ──
_register("radial_spin_blur_gpu",
          "Radial & Spin Blur (client-GPU twin of node 486)",
          "filter", _filter_typed('''
    // Motion-blur kernel: average samples laid along a radial (zoom) and
    // rotational (spin) path about a pivot. Continuous spin + a cos breathe
    // keep the live preview animated (no sin-phase 0/pi degeneracy).
    int n = int(u_length) + 1;
    n = clamp(n, 2, 32);
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 p = uv - ctr;
    float maxr = max(length(p * u_resolution), 1.0);
    float breathe = 1.0 + 0.3 * cos(u_time * u_anim_speed);
    float ang = u_time * u_anim_speed * 0.5;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 32; i++) {
        if (i >= n) break;
        float f = (float(i) / float(n - 1)) - 0.5;          // -0.5 .. 0.5
        float disp = (u_length / maxr) * f * breathe;
        vec2 q = ctr + p * (1.0 - disp);                    // radial zoom
        vec2 qr = ctr + rot(ang) * p;                       // rotational spin
        vec2 samp = mix(q, qr, 0.35);
        acc += sample(clamp(samp, 0.0, 1.0)).rgb;
    }
    acc /= float(n);
    f_color = vec4(acc, 1.0);
'''), uniforms={
    "length":     {"glsl": "float", "min": 0.0, "max": 64.0, "default": 14.0,
                   "description": "blur strength in px (edge displacement)"},
    "center_x":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "blur pivot x (0-1)"},
    "center_y":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "blur pivot y (0-1)"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0,
                   "description": "animation speed"},
})