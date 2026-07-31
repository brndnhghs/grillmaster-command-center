"""quasicrystal_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("quasicrystal_typed", "Quasicrystal interference with typed freq/rot/waves (node 253)",
          "procedural", _INFERNO_GPU + '''void main() {
    float freq = max(u_frequency, 0.005);
    float amp  = max(u_amplitude, 0.01);
    float rot  = u_rotation;
    int nwaves = int(clamp(u_waves, 2.0, 24.0));
    vec2 p = (v_uv - 0.5) * u_resolution;
    float t = u_time * 0.05;
    float sum = 0.0;
    for (int i = 0; i < 24; i++) {
        if (i >= nwaves) break;
        float fi = float(i);
        float a = rot + fi * 2.3999632 + t * 0.1;
        vec2 dir = vec2(cos(a), sin(a));
        sum += amp * sin(dot(p, dir) * freq * 0.01 + fi * 1.7);
    }
    float v = clamp(0.5 + 0.5 * (sum / float(max(nwaves, 1))), 0.0, 1.0);
    f_color = vec4(inferno(v), 1.0);
}
''', uniforms={
    "frequency": {"glsl": "float", "min": 0.5, "max": 10.0, "default": 3.0,
                  "description": "wave frequency"},
    "amplitude": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                  "description": "wave amplitude"},
    "rotation":  {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0,
                  "description": "global rotation (rad)"},
    "waves":     {"glsl": "int", "min": 2, "max": 24, "default": 12,
                  "description": "number of interfering waves"},
})