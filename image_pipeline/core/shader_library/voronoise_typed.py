"""voronoise_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register






# Typed-uniform twin of CPU node 528 (Voronoise). Closed-form f(uv,t): the
# Voronoi feature points orbit with u_time so the live preview is genuinely
# animated (survives the contrast-only static liveness cull). Every
# numeric node param is exposed as a named uniform; colormode/palette/anim_mode
# are choice strings (pitfall #14) left unmapped -> preview uses the inferno
# default (node 528's default colormode). CPU numpy node 528 stays authoritative
# for export. 528 is a CPU node id above 301.
_register("voronoise_typed",
          "IQ smooth Voronoi (voronoise) with typed scale/jitter/smoothness/octaves (node 528)",
          "procedural", '''vec2 vhash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

// Self-contained inferno colormap (node 528 default colormode).
vec3 vinferno(float t) {
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

// IQ-style smooth Voronoi: w blends nearest-distance cells (w=0) into a
// smoothly-averaged noise field (w=1); jitter displaces the cell points.
float voronoise(vec2 x, float w, float jitter, float t) {
    vec2 n = floor(x);
    vec2 f = fract(x);
    float f1 = 8.0, f2 = 8.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = vhash22(n + g);
            // Feature points orbit over time so the field is animated.
            vec2 disp = jitter * (0.5 + 0.5 * vec2(sin(t + 6.2831 * o.x),
                                                  cos(t + 6.2831 * o.y)));
            vec2 pnt = g + disp;
            float d = length(pnt - f);
            if (d < f1) { f2 = f1; f1 = d; }
            else if (d < f2) { f2 = d; }
        }
    }
    // w=0 -> crisp Voronoi cell distance; w=1 -> smooth averaged field.
    return mix(f1, 0.5 * (f1 + f2), w);
}

void main() {
    vec2 uv = v_uv * max(u_scale, 0.5);
    float t = u_time * 0.4;
    float v = 0.0, amp = 0.5, norm = 0.0;
    vec2 q = uv;
    for (int o = 0; o < 5; o++) {
        if (o >= int(u_octaves)) break;
        v += amp * voronoise(q, u_smoothness, u_jitter, t + float(o) * 1.3);
        norm += amp;
        q *= u_lacunarity;
        amp *= u_gain;
    }
    v = clamp(v / max(norm, 1e-3), 0.0, 1.0);
    v = clamp((v - 0.5) * u_contrast + 0.5, 0.0, 1.0);
    f_color = vec4(vinferno(v), 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 1.0, "max": 24.0, "default": 8.0,
                   "description": "grid frequency / zoom"},
    "jitter":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                   "description": "cell-point displacement (0=regular, 1=jittered)"},
    "smoothness": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                   "description": "metric: 1=averaged noise, 0=min-distance cells"},
    "octaves":    {"glsl": "float", "min": 1.0, "max": 5.0, "default": 1.0,
                   "description": "fBM octaves stacked for detail"},
    "lacunarity": {"glsl": "float", "min": 1.5, "max": 3.0, "default": 2.0,
                   "description": "frequency multiplier per octave"},
    "gain":       {"glsl": "float", "min": 0.3, "max": 0.8, "default": 0.5,
                   "description": "amplitude falloff per octave"},
    "contrast":   {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0,
                   "description": "tone contrast"},
})