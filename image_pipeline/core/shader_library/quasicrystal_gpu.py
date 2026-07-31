"""quasicrystal_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("quasicrystal_gpu", "Quasicrystal wave superposition (client-GPU twin of node 02)", "procedural", _INFERNO + """
float h11(float n){ return fract(sin(n*127.1)*43758.5453); }
void main() {
    // u_params.x = frequency, .y = amplitude, .z = rotation, .w = wave count
    float freq = max(u_frequency, 0.005);
    float amp  = (u_amplitude <= 0.0) ? 1.0 : u_amplitude;
    float rot  = u_rotation;
    int nwaves = int(clamp(u_waves, 2.0, 24.0));
    vec2 p = v_uv * u_resolution - 0.5 * u_resolution;      // centered pixel coords
    float phi = 3.14159265 * (1.0 + 2.2360679) / 2.0;        // pi*(1+sqrt5)/2 (penrose)
    float field = 0.0;
    for (int i = 0; i < 24; i++) {
        if (i >= nwaves) break;
        float fi = float(i);
        float theta = mod(fi * 6.2831853 / phi + rot, 6.2831853);
        float ph = h11(fi + 1.0) * 6.2831853;                // hash phase (!= numpy RNG)
        float f  = freq * (0.5 + h11(fi + 100.0));           // hash freq jitter
        float proj = p.x * cos(theta) + p.y * sin(theta);
        field += sin(proj * f + ph) * amp;
    }
    float result = field / float(nwaves) * 0.5 + 0.5;        // approx of CPU norm()
    f_color = vec4(inferno(clamp(result, 0.0, 1.0)), 1.0);
}
""",
    uniforms={
    "frequency": {"glsl": "float", "min": 0.005, "max": 2.0, "default": 0.5, "description": "wave frequency"},
    "amplitude": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 1.0, "description": "wave amplitude"},
    "rotation": {"glsl": "float", "min": 0.0, "max": 6.2831853, "default": 0.0, "description": "wave rotation"},
    "waves": {"glsl": "float", "min": 2.0, "max": 24.0, "default": 12.0, "description": "wave count"}
}
    )