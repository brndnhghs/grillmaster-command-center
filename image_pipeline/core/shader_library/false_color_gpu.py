"""false_color_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("false_color_gpu",
          "False-color IR remap (client-GPU twin of node 77)",
          "filter", _filter_typed('''
    // u_strength: 0 = grayscale, 1 = full false-color blend. The node's
    // color_scheme is a STRING choice (pitfall #14) so the preview locks to
    // the thermal ramp; the CPU fn stays authoritative for all schemes.
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float s = clamp(u_strength, 0.0, 1.0);
    int scheme = 1;   // preview default: thermal ramp

    vec3 heat;
    if (scheme == 1) {                      // thermal (black-red-yellow-white)
        heat = vec3(smoothstep(0.0, 0.4, lum),
                    smoothstep(0.3, 0.75, lum),
                    smoothstep(0.7, 1.0, lum));
    } else if (scheme == 2) {               // vegetation (brown -> green)
        heat = vec3(0.35 * (1.0 - lum), 0.15 + 0.7 * lum, 0.15 * (1.0 - lum) + 0.1 * lum);
    } else if (scheme == 3) {               // urban (blue-gray -> cyan -> magenta)
        heat = vec3(0.3 + 0.4 * lum, 0.4 + 0.3 * lum, 0.6 + 0.4 * sin(lum * 3.14159));
    } else {                                // standard IR ramp (inferno-like)
        heat = vec3(lum * 1.4, lum * lum * 1.2, (1.0 - lum) * 0.6 + lum * 0.2);
    }
    heat = clamp(heat, 0.0, 1.0);
    vec3 col = mix(vec3(lum), heat, s);
    f_color = vec4(col, 1.0);
'''), uniforms={
        "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "false-color blend"},
    })