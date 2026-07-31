"""chladni_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("chladni_typed", "Chladni nodal plate with typed mode/rotation/phase (node 251)",
          "procedural", _INFERNO_GPU + '''void main() {
    float m = max(u_m_mode, 0.5);
    float n = max(u_n_mode, 0.5);
    float rot_ang = u_rotation;
    float ph = u_phase;
    vec2 p = (v_uv - 0.5) * 2.0;
    vec2 pr = rot(rot_ang) * p;
    float u = sin(m * 3.14159265 * (pr.x + 1.0) * 0.5 + ph)
            * sin(n * 3.14159265 * (pr.y + 1.0) * 0.5 + ph);
    float sig = tanh(u * u_contrast * 4.0);
    float v = 0.5 + 0.5 * sig;
    f_color = vec4(inferno(v), 1.0);
}
''', uniforms={
    "m_mode":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
                  "description": "x mode number"},
    "n_mode":    {"glsl": "float", "min": 1.0, "max": 12.0, "default": 3.0,
                  "description": "y mode number"},
    "rotation":  {"glsl": "float", "min": -3.14159, "max": 3.14159, "default": 0.0,
                  "description": "plate rotation (rad)"},
    "phase":     {"glsl": "float", "min": -3.14159, "max": 3.14159, "default": 0.0,
                  "description": "phase shimmer (rad)"},
    "contrast":  {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.0,
                  "description": "nodal-line sharpness"},
})