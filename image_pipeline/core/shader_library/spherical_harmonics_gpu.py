"""spherical_harmonics_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




_register("spherical_harmonics_gpu",
          "Spherical harmonics banding (client-GPU twin of node 104)",
          "procedural", '''
void main() {
    // Named uniforms (auto-declared) match node 104's real params; the client
    // reads them by name (pitfall #14b). Closed-form approximation of
    // spherical-harmonic Y_l^m banding projected to a 2D (theta, phi) map.
    float L = max(1.0, floor(u_max_l + 0.5));
    float th = v_uv.y * 3.14159265;     // polar angle 0..pi
    float ph = v_uv.x * 6.2831853;      // azimuth 0..2pi
    float t = u_time * u_anim_speed * 0.15;

    float f = 0.0;
    float wsum = 0.0;
    for (int li = 1; li <= 8; li++) {
        if (float(li) > L) break;
        float fl = float(li);
        float Pl = cos(fl * th);        // meridional banding (Legendre-like)
        for (int mi = 0; mi <= 8; mi++) {
            if (float(mi) > fl) break;
            float fm = float(mi);
            float az = cos(fm * ph + (fm == 0.0 ? 0.0 : t) + fm * 1.3);
            float w = 1.0 / (fl + 1.0);
            f += Pl * az * w;
            wsum += w;
        }
    }
    f /= max(wsum, 1.0);

    // Twist: azimuthal phase shear (node twist_wave character).
    float twist = sin(ph * (1.0 + u_twist_amplitude) + t * 2.0) * u_osc_spread * 0.15;
    f += twist;

    float val = clamp(f * 0.5 + 0.5, 0.0, 1.0);
    val = pow(val, 1.0 / max(u_glow_strength, 0.2));   // glow emphasis
    float gray = clamp(val * u_amplitude, 0.0, 1.0);
    f_color = vec4(vec3(gray), 1.0);
}
''',
    uniforms={
    "max_l": {"glsl": "float", "min": 1.0, "max": 8.0, "default": 5.0, "description": "max shell l"},
    "amplitude": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.5, "description": "brightness amplitude"},
    "glow_strength": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.5, "description": "glow emphasis"},
    "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
    "twist_amplitude": {"glsl": "float", "min": 0.5, "max": 5.0, "default": 2.0, "description": "twist amplitude"},
    "osc_spread": {"glsl": "float", "min": 0.0, "max": 5.0, "default": 1.5, "description": "oscillator spread"}
    }
    )