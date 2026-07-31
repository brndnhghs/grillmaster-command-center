"""galaxy_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── GPU-First gap mirrors: closed-form f(uv,t) twins for CPU nodes 487/441/108 ──
# Each CPU node is a per-pixel closed-form generator with NO close existing
# twin, so it gets a brand-new GLSL twin wired via a typed-uniform
# CLIENT_GPU_SHIMS entry (gpu_shaders.py). Every numeric CPU param becomes a
# named u_<name> uniform/SCALAR port (typed-uniform contract). CPU fns stay
# authoritative for export; GPU live preview is approximate by design. Palettes
# are inlined (cosine / hsv) to avoid the late-helper (inferno) ordering pitfall.

_register("galaxy_gpu", "Galaxy Generator (client-GPU twin of node 487)", "procedural",
'''void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    float incl = mix(0.45, 1.0, clamp(u_inclination, 0.0, 1.0));
    uv.y /= incl;
    float r = length(uv);
    float ang = atan(uv.y, uv.x);
    float t = u_time * 0.2 * (0.3 + u_rotation_speed);

    // central bulge
    float bulge = exp(-r * r / (u_bulge_size * u_bulge_size)) * 1.2;

    // logarithmic spiral arms
    float spiral = 0.0;
    int N = int(clamp(u_arms, 1.0, 6.0));
    for (int i = 0; i < 6; i++) {
        if (i >= N) break;
        float fi = float(i);
        float a = ang + log(r + 0.06) / max(u_tightness, 0.05)
                  - fi * 6.2831853 / float(N) + t;
        float c = cos(a);
        spiral += exp((c - 1.0) * (4.0 / max(u_arm_spread, 0.02)));
    }
    spiral *= exp(-r * 1.3);

    float density = bulge + spiral;

    // procedural star sparkle
    float tw = hash21(floor(uv * 90.0));
    density += smoothstep(0.93, 1.0, tw) * (0.3 + 0.7 * fbm(uv * 18.0 + t)) * (0.4 + density);
    density = clamp(density * u_brightness, 0.0, 1.0);

    // natural palette: warm core -> blue rim, depth by radius
    vec3 core = vec3(1.0, 0.85, 0.6);
    vec3 midc = vec3(1.0, 0.95, 0.85);
    vec3 rim  = vec3(0.55, 0.7, 1.0);
    vec3 col = mix(core, midc, smoothstep(0.0, 0.3, r));
    col = mix(col, rim, smoothstep(0.3, 0.9, r));
    col *= density;
    col += vec3(0.02, 0.03, 0.05) * (1.0 - r);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "arms": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 2.0, "description": "number of spiral arms"},
    "tightness": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.5, "description": "spiral winding tightness"},
    "arm_spread": {"glsl": "float", "min": 0.02, "max": 0.4, "default": 0.15, "description": "arm width"},
    "bulge_size": {"glsl": "float", "min": 0.05, "max": 0.5, "default": 0.2, "description": "central bulge radius"},
    "inclination": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.3, "description": "view inclination (vertical squash)"},
    "rotation_speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "arm rotation speed"},
    "brightness": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0, "description": "overall brightness"},
})