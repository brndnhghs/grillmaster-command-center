"""kaleidoscope_mirror_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("kaleidoscope_mirror_gpu", "Kaleidoscope Mirror (client-GPU twin of node 460)", "filter", _filter_typed('''
    vec2 ctr = vec2(u_center_x, u_center_y);
    vec2 p = uv - ctr;
    vec2 wp = p;
    if (u_warp_amount > 0.001) {
        float w = u_warp_amount * 0.3;
        float nx = fbm(p * u_warp_scale + vec2(u_time * 0.1, 0.0)) - 0.5;
        float ny = fbm(p * u_warp_scale + vec2(17.0, u_time * 0.1)) - 0.5;
        wp = p + w * vec2(nx, ny);
    }
    float a = atan(wp.y, wp.x) + radians(u_rotation);
    float r = length(wp) * u_r_scale;
    float seg = 6.2831853 / max(3.0, u_segments);
    a = mod(a, seg);
    if (u_mirror > 0.5) {
        a = abs(a - seg * 0.5);
    }
    vec2 sp = vec2(cos(a), sin(a)) * r + ctr;
    f_color = texture(u_texture, sp);
'''), uniforms={
    "segments": {"glsl": "float", "min": 3.0, "max": 24.0, "default": 8.0, "description": "mirror wedges (dihedral symmetry order)"},
    "center_x": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "symmetry center X (0-1)"},
    "center_y": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "symmetry center Y (0-1)"},
    "rotation": {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0, "description": "pattern rotation in degrees"},
    "r_scale": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "radial zoom into the wedge"},
    "mirror": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "mirror-fold adjacent wedges (dihedral)"},
    "warp_amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "fBm domain-warp strength (0 = pure geometry)"},
    "warp_scale": {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0, "description": "domain-warp noise frequency"},
})