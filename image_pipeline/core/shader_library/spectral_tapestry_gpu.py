"""spectral_tapestry_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spectral_tapestry_gpu",
          "Spectral tapestry interference (client-GPU twin of node 161)",
          "procedural", '''
void main() {
    // Named uniforms match node 161's real params; client reads by name
    // (pitfall #14b). Closed-form approximation of the spectral-PDE field: a
    // golden-angle fan of drifting sinusoidal gratings, contrast-shaped by
    // coupling. The CPU spectral simulation stays authoritative for export.
    float nm = clamp(floor(u_n_modes), 6.0, 40.0);
    float c = u_coupling;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;
    float scale = 6.2831853 / max(res.x, res.y);
    float t = u_time * (0.05 + u_drift_speed * 20.0);

    float acc = 0.0;
    for (int i = 0; i < 40; i++) {
        if (float(i) >= nm) break;
        float k = float(i);
        float ang = k * 2.39996323;           // golden angle fan
        vec2 d = vec2(cos(ang), sin(ang));
        float w = 3.0 + k * (1.0 + c * 2.0);  // finer modes with coupling
        float ph = t * (0.5 + 0.05 * k) + k * 1.7;
        acc += sin(dot(p, d) * scale * w + ph);
    }
    acc /= max(nm, 1.0);

    float val = acc * 0.5 + 0.5;
    // coupling sharpens the interference (storm-like thresholding)
    val = clamp(mix(val, smoothstep(0.35, 0.65, val), clamp(c * 0.5, 0.0, 1.0)), 0.0, 1.0);
    val += (noise(p * 0.05 + vec2(t)) - 0.5) * u_noise * 4.0;   // stochastic grain
    val = clamp(val, 0.0, 1.0);
    f_color = vec4(vec3(val), 1.0);
}
''',
    uniforms={
    "n_modes": {"glsl": "float", "min": 8.0, "max": 80.0, "default": 25.0, "description": "mode count"},
    "coupling": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.4, "description": "mode coupling"},
    "drift_speed": {"glsl": "float", "min": 0.0, "max": 0.05, "default": 0.005, "description": "drift speed"},
    "noise": {"glsl": "float", "min": 0.0, "max": 0.1, "default": 0.01, "description": "stochastic noise"}
    }
    )