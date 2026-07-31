"""metaballs_505_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── 505 Metaballs (Procedural) ──
_register("metaballs_505_gpu", "Metaballs (client-GPU twin of node 505)", "procedural", '''
void main() {
    vec2 uv = v_uv;
    float t = u_time * u_anim_speed;
    int N = int(clamp(u_balls, 2.0, 16.0));
    float field = 0.0;
    for (int i = 0; i < 16; i++) {
        if (i >= N) break;
        float fi = float(i);
        float ang = fi * 2.3999632 + t * (0.5 + 0.1 * fi);   // golden-angle spread
        float rad = u_drift_amp * (0.5 + 0.5 * sin(t * 0.7 + fi));
        vec2 c = vec2(0.5) + rad * vec2(cos(ang), sin(ang))
                 + 0.10 * vec2(sin(t + fi), cos(t * 1.1 + fi * 2.0));
        vec2 d = uv - c;
        float bs = u_ball_size * (1.0 + 0.25 * sin(t + fi));  // pulse
        field += (bs * bs) / max(dot(d, d), 1e-4);
    }
    float m = smoothstep(u_threshold - u_edge_soft, u_threshold + u_edge_soft, field);
    float hue = fract(field * 0.15 + t * 0.05);
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
    col *= m;
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "balls": {"glsl": "float", "min": 2.0, "max": 16.0, "default": 8.0, "description": "number of metaballs"},
        "ball_size": {"glsl": "float", "min": 0.02, "max": 0.3, "default": 0.1, "description": "ball radius (frac of canvas)"},
        "threshold": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "iso-level"},
        "edge_soft": {"glsl": "float", "min": 0.0, "max": 0.6, "default": 0.15, "description": "edge softness"},
        "drift_amp": {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.12, "description": "orbit amplitude"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed"},
    })