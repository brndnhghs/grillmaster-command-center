"""wave_pattern_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("wave_pattern_gpu", "Periodic wave stripes: sine/triangle/square/saw, typed controls",
          "procedural", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float a = radians(u_angle);
    float x = dot(uv, vec2(cos(a), sin(a))) * u_frequency + u_time * u_phase_speed;
    float ph = fract(x);
    float w;
    if (u_waveform == 1)      w = 1.0 - abs(ph * 2.0 - 1.0);          // triangle
    else if (u_waveform == 2) w = step(ph, clamp(u_duty, 0.01, 0.99)); // square
    else if (u_waveform == 3) w = ph;                                  // saw
    else                      w = 0.5 + 0.5 * sin(ph * 6.28318530);    // sine
    f_color = vec4(mix(u_color_a, u_color_b, w), 1.0);
}
''', uniforms={
    "waveform":    {"glsl": "choice", "choices": ["sine", "triangle", "square", "saw"],
                    "default": "sine", "description": "wave shape"},
    "frequency":   {"glsl": "float", "min": 0.5, "max": 64.0, "default": 8.0,
                    "description": "stripe frequency"},
    "angle":       {"glsl": "float", "min": 0.0, "max": 360.0, "default": 45.0,
                    "description": "stripe angle (deg)"},
    "phase_speed": {"glsl": "float", "min": -4.0, "max": 4.0, "default": 0.5,
                    "description": "phase drift speed (per second)"},
    "duty":        {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                    "description": "duty cycle (square wave)"},
    "color_a":     {"glsl": "color", "default": "#0b1026", "description": "trough color"},
    "color_b":     {"glsl": "color", "default": "#ff9d2e", "description": "crest color"},
})