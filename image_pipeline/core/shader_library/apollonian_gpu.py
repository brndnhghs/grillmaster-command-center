"""apollonian_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("apollonian_gpu",
          "IQ Apollonian gasket via iterated fold+inversion; typed depth/seed_curv mirror node 514",
          "procedural", '''// Self-contained inferno colormap (matches other typed twins).
vec3 apinferno(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.00021894, 0.00165100, -0.01948090);
    const vec3 c1 = vec3(0.10651342, 0.56395644,  3.93271239);
    const vec3 c2 = vec3(11.60249308, -3.97285397, -15.94239411);
    const vec3 c3 = vec3(-41.70399613, 17.43639888, 44.35414520);
    const vec3 c4 = vec3(77.16293570, -33.40235894, -81.80730926);
    const vec3 c5 = vec3(-71.31942824, 32.62606426, 73.20951986);
    const vec3 c6 = vec3(25.13112622, -12.24266895, -23.07032500);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// IQ "Apollonian" recurrence: repeatedly wrap into [-1,1] then invert about the
// origin. `u_depth` mirrors node 514's recursion depth (nested-circle density);
// `u_seed_curv` mirrors the seed-circle bend (curvature per inversion). `t`
// slowly rotates the frame so the packing animates when a driver is wired.
void main() {
    vec2 p = (v_uv - 0.5) * 2.4;
    float t = u_time * 0.15;
    float ca = cos(t), sa = sin(t);
    p = mat2(ca, -sa, sa, ca) * p;
    float scale = 1.0;
    float d = 1e9;
    float fold = 0.55 + u_seed_curv * 0.28;   // seed bend -> inversion scale
    for (int i = 0; i < 8; i++) {
        if (i >= int(u_depth)) break;
        p = -1.0 + 2.0 * fract(0.5 * p + 0.5);
        float r2 = dot(p, p);
        float k = fold / max(r2, 1e-4);
        p *= k;
        scale *= k;
        d = min(d, abs(p.y) / scale);
    }
    float v = 1.0 - clamp(pow(d * 6.0, 0.35), 0.0, 1.0);
    v = clamp((v - 0.5) * 1.4 + 0.5, 0.0, 1.0);
    f_color = vec4(apinferno(v), 1.0);
}
''', uniforms={
    "depth":     {"glsl": "float", "min": 1.0, "max": 6.0, "default": 4.0,
                  "description": "recursion depth (denser nested circles)"},
    "seed_curv": {"glsl": "float", "min": 1.0, "max": 4.0, "default": 2.0,
                  "description": "seed-circle bend -> inversion curvature"},
})