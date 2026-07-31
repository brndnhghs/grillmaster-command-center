"""gyroid_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# ── Typed closed-form pattern node (id 301) ──────────────────────────────
# Gyroid / triply-periodic minimal surface — a 2D slice of the 3D field
#   g(x,y,z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
# popularized in shader art (Inigo Quilez, ~2010s). The viewing plane is spun
# in-plane and the slice depth advances with time, so the woven surface appears
# to rotate and flow through 3D space. Closed-form f(uv,t) → exact GPU/CPU
# parity preview (no seeded-layout divergence). Additive typed-uniform layer.
_register("gyroid_typed", "Gyroid / triply-periodic minimal-surface slice (typed, node 301)",
          "procedural", _INFERNO_GPU + '''void main() {
    // 2D slab of the 3D gyroid scalar field. Spinning the plane in-plane and
    // advancing the slice depth with time makes the woven surface appear to
    // rotate and flow through 3D space.
    vec2 p = (v_uv - 0.5) * u_scale;
    float t = u_time;
    p = rot(t * u_spin) * p;                // in-plane spin
    float z = u_slice + t * u_anim_speed;   // advance the slice through volume
    float g = sin(p.x) * cos(p.y) + sin(p.y) * cos(z) + sin(z) * cos(p.x);
    float v = 0.5 + 0.5 * sin(g * 1.3);     // colormap coordinate
    vec3 base = inferno(clamp(v, 0.0, 1.0));
    float wall = smoothstep(u_thickness, u_thickness * 0.25, abs(g));
    vec3 col = mix(base, u_wall, wall);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 4.0, "max": 40.0, "default": 14.0,
                   "description": "feature density (cells across view)"},
    "thickness":  {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.12,
                   "description": "gyroid wall thickness"},
    "slice":      {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0,
                   "description": "static slice depth offset (rad)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                   "description": "slice advance speed through volume"},
    "spin":       {"glsl": "float", "min": 0.0, "max": 6.0, "default": 0.6,
                   "description": "in-plane rotation speed"},
    "wall":       {"glsl": "color", "default": "#ff5cf0",
                   "description": "wall highlight color"},
})