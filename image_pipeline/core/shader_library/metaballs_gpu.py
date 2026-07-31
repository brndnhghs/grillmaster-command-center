"""metaballs_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _inferno_local



_register("metaballs_gpu",
          "Metaballs isosurface field (client-GPU twin of node 53)",
          "procedural", _inferno_local('') + '''
void main() {
    // u_params.x = isovalue (0.5 -> ~0.425), u_params.y = ball_speed (0.5 -> ~2.55).
    float iso   = 0.05 + u_isovalue * 0.75;
    float speed = 0.1  + u_ball_speed * 4.9;
    float t = u_time * 0.05 * speed;

    // Closed-form soft metaball field from N orbiting balls (pure f(uv, t)).
    vec2 p = v_uv;
    float field = 0.0;
    const int N = 14;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float ang = fi * 2.399963;                 // golden-angle spread
        float orbit = 0.18 + 0.16 * hash21(vec2(fi, 1.7));
        float wx = 0.5 + orbit * cos(t * (0.6 + 0.05 * fi) + ang);
        float wy = 0.5 + orbit * sin(t * (0.6 + 0.05 * fi) + ang * 1.3);
        vec2 c = vec2(wx, wy);
        float ri = 0.06 + 0.05 * hash21(vec2(fi, 9.1));
        float d2 = dot(p - c, p - c);
        field += (ri * ri) / (ri * ri + d2 + 1e-4);
    }
    float f = clamp(field * 0.5, 0.0, 1.0);
    vec3 col = inferno(f);
    // Bright isosurface edge near the threshold.
    float edge = smoothstep(iso - 0.04, iso, field * (iso + 0.2))
               * (1.0 - smoothstep(iso, iso + 0.04, field * (iso + 0.2)));
    col += edge * 0.35;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''',
    uniforms={
    "isovalue": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "isosurface value"},
    "ball_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "ball orbit speed"}
    }
    )