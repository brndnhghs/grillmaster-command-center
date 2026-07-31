"""lyapunov_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _FRACTAL_HELPERS



_register("lyapunov_gpu", "Lyapunov exponent map (client-GPU twin of node 69)", "procedural",
          _FRACTAL_HELPERS + '''
void main() {
    vec2 uv = v_uv;
    // r_min / r_max are the logistic r-range (match node 69's real params);
    // the A/B perturbation sequence and warmup/measure counts are choice/int
    // controls (pitfall #14) left unmapped — the twin renders the default ABAB map.
    vec2 rmin = vec2(u_r_min, u_r_min);
    vec2 rmax = vec2(u_r_max, u_r_max);
    vec2 r = mix(rmin, rmax, uv);
    // Logistic-map A/B perturbation (ABAB...), 8 chars.
    float lambda = 0.0;
    float x = 0.5;
    const float WARM = 30.0;
    const float MEAS = 80.0;
    for (float i = 0.0; i < (WARM + MEAS); i += 1.0) {
        int k = int(mod(i, 8.0));
        float rk = (k == 0 || k == 2 || k == 4 || k == 6) ? r.x : r.y;
        float deriv = rk * (1.0 - 2.0 * x);
        x = rk * x * (1.0 - x);
        if (i >= WARM) {
            lambda += log(abs(deriv) + 1e-8);
        }
    }
    lambda = lambda / MEAS;
    float t = clamp(0.5 + 0.5 * lambda / 2.0, 0.0, 1.0);
    t = t;
    f_color = vec4(fractal_palette(t), 1.0);
}
''',
    uniforms={
    "r_min": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "lower logistic r"},
    "r_max": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "upper logistic r"}
}
    )