"""rings_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("rings_gpu", "Concentric animated rings (typed procedural)",
          "procedural", '''
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float r = length(uv) * u_freq;
    float t = u_time * u_speed;
    float w = 0.5 + 0.5 * sin(r * 6.28318530 - t * 2.0);
    w = pow(w, max(u_sharp, 0.05));
    vec3 col = mix(u_color_a, u_color_b, w);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":  {"glsl": "float", "min": 1.0, "max": 40.0, "default": 10.0,
              "description": "ring frequency"},
    "speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
              "description": "animation speed"},
    "sharp": {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "ring edge sharpness"},
    "color_a": {"glsl": "color", "default": "#05070f", "description": "trough color"},
    "color_b": {"glsl": "color", "default": "#4de0ff", "description": "crest color"},
})