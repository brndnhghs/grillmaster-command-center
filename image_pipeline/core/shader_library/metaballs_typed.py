"""metaballs_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("metaballs_typed", "Metaballs isosurface field with typed isovalue/speed/colors (node 254)",
          "procedural", _INFERNO_GPU + '''void main() {
    float iso   = 0.05 + u_isovalue * 0.75;
    float speed = 0.1  + u_ball_speed * 4.9;
    float t = u_time * 0.05 * speed;
    vec2 p = v_uv;
    float field = 0.0;
    const int N = 14;
    for (int i = 0; i < N; i++) {
        float fi = float(i);
        float ang = fi * 2.399963;
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
    float edge = smoothstep(iso - 0.04, iso, field * (iso + 0.2))
               * (1.0 - smoothstep(iso, iso + 0.04, field * (iso + 0.2)));
    col += edge * 0.35;
    col = mix(col, u_tint, u_tint_strength);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "isovalue":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "iso threshold"},
    "ball_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "orbit speed"},
    "tint":       {"glsl": "color", "default": "#ffffff", "description": "edge tint"},
    "tint_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                      "description": "edge tint strength"},
})