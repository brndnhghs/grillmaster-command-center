"""aurora_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── GPU-First gap mirrors: closed-form f(uv,t) twins for CPU nodes 523/954/512 ──
# These three CPU nodes are per-pixel closed-form fields with NO close existing
# twin (Aurora Borealis, Autostereogram, SIREN Field). Each maps to a client-GPU
# shim in image_pipeline/methods/gpu_shaders.py (typed-uniform contract: every
# numeric CPU param becomes a named u_<name> uniform, bound to a real SCALAR
# port). CPU fns stay authoritative for export; parity is approximate by design.
# No late helpers (inferno/hsv2rgb) are used — a local cosine palette is inlined
# to avoid the late-helper ordering pitfall.

_register("aurora_gpu", "Aurora Borealis (client-GPU twin of node 523)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float t = u_time;
    vec3 col = vec3(0.01, 0.02, 0.06) * (1.0 - 0.4 * uv.y);
    // starfield
    vec2 sg = floor(uv * 220.0);
    float h = hash21(sg);
    if (h > 1.0 - clamp(u_star_density, 0.0, 1.0) * 0.5) {
        float d = length(fract(uv * 220.0) - 0.5);
        col += vec3(smoothstep(0.12, 0.0, d)) * (0.4 + 0.6 * hash21(sg + 2.1));
    }
    int N = int(clamp(u_curtain_count, 1.0, 8.0));
    for (int i = 0; i < 8; i++) {
        if (i >= N) break;
        float fi = float(i);
        float baseY = -0.45 + fi * 0.22;
        float wob = fbm(vec2(uv.x * u_turbulence + t * u_drift_speed * 0.25 + fi * 3.1, t * 0.12));
        float yc = baseY + (wob - 0.5) * 0.7 + 0.12 * sin(t * 0.3 + fi);
        float dist = abs(uv.y - yc);
        float width = 0.16 * (0.6 + 0.6 * wob);
        float beam = exp(-dist * dist / (width * width));
        float streak = 0.6 + 0.4 * sin(uv.x * 38.0 + t * u_drift_speed * 2.0 + fi * 5.0);
        beam *= streak;
        vec3 ac = mix(vec3(0.1, 1.0, 0.45), vec3(1.0, 0.25, 0.35),
                      clamp(u_red_fringe, 0.0, 1.0) * abs(uv.x));
        ac = mix(ac, vec3(0.35, 0.75, 1.0), 0.25 * sin(fi * 1.7));
        col += ac * beam * u_intensity * (0.35 + 0.65 * fbm(vec2(uv.x * 3.0 - t * 0.2, fi)));
    }
    // vertical extent mask driven by beam_height
    col *= smoothstep(u_beam_height + 0.4, u_beam_height - 0.4, max(uv.y, -1.0));
    col += vec3(0.05, 0.0, 0.08) * u_color_shift;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "curtain_count": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 4.0, "description": "number of aurora curtains"},
    "drift_speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "horizontal drift speed"},
    "intensity": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.0, "description": "overall brightness"},
    "beam_height": {"glsl": "float", "min": 0.2, "max": 0.9, "default": 0.6, "description": "vertical extent of curtains"},
    "color_shift": {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "palette color offset"},
    "turbulence": {"glsl": "float", "min": 0.5, "max": 6.0, "default": 2.5, "description": "fbm turbulence frequency"},
    "star_density": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.35, "description": "starfield density"},
    "red_fringe": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "red fringe amount"},
})