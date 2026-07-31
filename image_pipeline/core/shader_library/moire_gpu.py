"""moire_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("moire_gpu",
          "Moiré interference (client-GPU twin of node 164)",
          "procedural", '''
void main() {
    // u_params.x = mode (0=radial,1=linear,2=spiral,3=hex),
    // u_params.y = speed1 (0.5 -> ~1.0), u_params.z = speed2 (0.5 -> ~1.28),
    // u_params.w = frequency (0.5 -> 20).
    int mode = int(floor(u_mode * 3.999));
    float s1 = 0.1 + u_speed1 * 1.9;      // ~1.0 at default
    float s2 = 0.1 + u_speed2 * 1.9;      // ~1.28 at default
    float freq = 5.0 + u_frequency * 45.0;   // 20 at default
    float t = u_time * 0.05;               // matches node: t = fr*0.05

    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;           // pixel-centered coords
    float scale = 1.0 / max(res.x, res.y) * 2.0 * 3.14159265;

    float g1, g2;
    float a1 = s1 * t, a2 = s2 * t;
    if (mode == 1) {                       // linear gratings
        g1 = 0.5 + 0.5 * sin(freq * (p.x * cos(a1) + p.y * sin(a1)) * scale);
        g2 = 0.5 + 0.5 * sin(freq * (p.x * cos(a2) + p.y * sin(a2)) * scale);
    } else if (mode == 2) {                // spiral
        float r = length(p);
        float th = atan(p.y, p.x);
        g1 = 0.5 + 0.5 * sin(freq * (r * scale + th / 6.2831853) * 6.2831853 + a1);
        g2 = 0.5 + 0.5 * sin(freq * (r * scale + th / 6.2831853) * 6.2831853 + a2);
    } else if (mode == 3) {                // hex (3-grating sum)
        float acc = 0.0;
        acc += 0.5 + 0.5 * sin(freq * (p.x) * scale + s1 * t);
        acc += 0.5 + 0.5 * sin(freq * (p.x * cos(1.0471975) + p.y * sin(1.0471975)) * scale + (s1 + 0.3) * t);
        acc += 0.5 + 0.5 * sin(freq * (p.x * cos(2.0943951) + p.y * sin(2.0943951)) * scale + (s1 + 0.6) * t);
        acc = clamp(acc / 3.0, 0.0, 1.0);
        f_color = vec4(vec3(acc), 1.0);
        return;
    } else {                               // radial (default)
        float r = length(p);
        g1 = 0.5 + 0.5 * sin(freq * r * scale);
        g2 = 0.5 + 0.5 * sin(freq * r * scale + a2);
    }
    float g = clamp(g1 * g2 * 2.0, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''',
    uniforms={
    "mode": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0, "description": "grating mode (0-3)"},
    "speed1": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "speed 1"},
    "speed2": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "speed 2"},
    "frequency": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "grating frequency"}
}
    )