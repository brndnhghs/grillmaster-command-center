"""apollonian_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# 305 — Apollonian gasket: iterate circle inversions in three mutually tangent
# circles inscribed in the unit disk. The limit set is the gasket.
_register("apollonian_typed", "Apollonian gasket via circle inversions (typed, node 305)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time * u_speed;
    p = rot(t * 0.4) * p;
    // three mutually-tangent unit-ish circles in the disk
    vec2 c0 = vec2(-0.5, 0.0);
    vec2 c1 = vec2( 0.5, 0.0);
    vec2 c2 = vec2( 0.0, 0.8660254);
    float r = 0.5;
    for (int i = 0; i < 8; i++) {
        vec2 c = (i % 3 == 0) ? c0 : (i % 3 == 1) ? c1 : c2;
        vec2 d = p - c;
        float d2 = max(dot(d, d), 1e-6);
        p = c + (r * r / d2) * d;
    }
    // Band the final inversion radius so the limit-set structure fills the full
    // inferno range (the previous clamp(length*0.5) saturated most of the disk
    // to white, hiding the u_scale/u_speed response — a silent no-op live
    // preview caught by test_typed_uniforms_drive_output). The +t*2.0 phase
    // keeps the rotation/animation visible (radius alone is rotation-invariant).
    float v = fract(length(p) * 0.5);
    v = 0.5 + 0.5 * sin(v * 6.2831853 * u_scale + t * 2.0);
    vec3 col = inferno(v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale": {"glsl": "float", "min": 2.0, "max": 12.0, "default": 5.0, "description": "view scale"},
    "speed": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "rotation speed"},
})