"""hex_mosaic_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Kaleidoscope Mirror (typed-uniform filter twin of CPU node 460) ──
# Dihedral mirror-fold: map every point into one wedge around a movable
# center, optionally reflect for true dihedral symmetry, apply radial zoom and
# a slow fBm domain-warp. Named typed uniforms mirror node 460's REAL numeric
# params (segments/center_x/center_y/rotation/r_scale/mirror/warp_amount/
# warp_scale) so LFO/counter drivers can modulate the live preview (the
# electrical-engineering trap: a contrast-only static clip is avoided because
# the wrap is genuine per-pixel motion). CPU node 460 stays authoritative;
# this is the live-preview path only. `step` is the prologue-reserved vec2, so
# the warp accumulator uses `q`/`wp` locals (pitfall #15b).
# ══════════════════════════════════════════════════════════════════════════
#  P0.7 closed-form pattern twins (gap nodes 466 / 505 / 426)
#  Live-preview GPU twins for pattern nodes whose CPU algorithm is a faithful
#  closed-form f(uv,t). CPU node stays authoritative for export; these drive the
#  client-side live preview. Choice params (anim_mode / color_mode / orientation
#  / source / bg / palette ...) are intentionally omitted — the twins animate
#  continuously from u_time so the preview is always live; the CPU export honours
#  the exact choice. Helper functions are inlined (only _PROLOGUE helpers +
#  u_time / u_resolution are used) to avoid the late-helper ordering pitfall.
# ══════════════════════════════════════════════════════════════════════════

# ── 466 Hexagonal Mosaic ──
_register("hex_mosaic_gpu", "Hexagonal Mosaic (client-GPU twin of node 466)", "procedural", '''
float hexDist(vec2 p) {
    p = abs(p);
    return max(dot(p, vec2(0.8660254, 0.5)), p.x);
}
void main() {
    float scale = max(4.0, u_hex_size);
    float latRot = u_rotation + u_time * 0.12 * u_anim_speed;
    scale *= 1.0 + 0.15 * sin(u_time * u_anim_speed);   // gentle breathe
    vec2 uv = (v_uv - 0.5) * u_resolution;
    uv = rot(latRot) * uv;                              // rot() is the prologue helper
    vec2 rr = vec2(1.0, 1.7320508) * scale;
    vec2 h = rr * 0.5;
    vec2 a = mod(uv, rr) - h;
    vec2 b = mod(uv + h, rr) - h;
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;
    vec2 cellId = uv - gv;
    float hexR = 0.8660254 * scale;                     // center-to-edge
    float edge = smoothstep(hexR * (1.0 - u_grout), hexR, hexDist(gv));
    float hsh = hash21(cellId * 0.013);
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (hsh + vec3(0.0, 0.33, 0.67)));
    col = mix(col, vec3(u_grout_color), edge);
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "hex_size": {"glsl": "float", "min": 4.0, "max": 60.0, "default": 18.0, "description": "hex cell radius (px)"},
        "rotation": {"glsl": "float", "min": 0.0, "max": 6.2832, "default": 0.0, "description": "lattice rotation (rad)"},
        "grout": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.12, "description": "grout line width"},
        "grout_color": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "grout grayscale"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed"},
    })