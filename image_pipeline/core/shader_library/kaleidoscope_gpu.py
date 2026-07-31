"""kaleidoscope_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("kaleidoscope_gpu", "Kaleidoscope mirror of the input image (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = radians(u_angle) + u_time * u_spin;
    uv = rot(a) * uv;
    float seg = max(float(u_segments), 2.0);
    float ang = atan(uv.y, uv.x);
    float rad = length(uv);
    // Fold angle into one wedge, then mirror within the wedge.
    float wedge = 6.28318530 / seg;
    ang = mod(ang, wedge);
    ang = abs(ang - wedge * 0.5);
    vec2 p = vec2(cos(ang), sin(ang)) * rad + 0.5;
    vec3 src = texture(u_texture, fract(p)).rgb;
    f_color = vec4(mix(src, src * (0.6 + 0.8 * u_zoom), u_zoom), 1.0);
}
''', uniforms={
    "segments": {"glsl": "int", "min": 2, "max": 24, "default": 6,
                 "description": "mirror segments"},
    "angle":    {"glsl": "float", "min": 0.0, "max": 360.0, "default": 0.0,
                 "description": "base rotation (deg)"},
    "spin":     {"glsl": "float", "min": -2.0, "max": 2.0, "default": 0.0,
                 "description": "auto-rotation speed"},
    "zoom":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                 "description": "center zoom"},
})