"""magnetic_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("magnetic_typed", "Magnetic dipole field: field-line ribbons (typed, node 286)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    // Distance to a dipole at origin; field strength falls as 1/r^3 inside.
    float r = max(length(p), 0.04);
    float pa = atan(p.y, p.x) + t;
    // Dipole potential ~ cos^2(theta) - 0.5 ; draw iso-lines of it.
    float pot = cos(pa) * cos(pa) - 0.5;
    float bands = sin(pot * u_lines * 6.28318530 / max(r, 0.04) * u_tight);
    float line = smoothstep(1.0 - u_sharp, 1.0, abs(bands));
    vec3 col = mix(u_bg, inferno(clamp(1.0 - r / (u_scale + 1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "lines":  {"glsl": "float", "min": 2.0, "max": 40.0, "default": 14.0,
               "description": "field-line count"},
    "tight":  {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
               "description": "line tightness"},
    "sharp":  {"glsl": "float", "min": 0.05, "max": 0.9, "default": 0.4,
               "description": "line sharpness"},
    "scale":  {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.9,
               "description": "field radius"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "rotation speed"},
    "bg":     {"glsl": "color", "default": "#04060c", "description": "background"},
})