"""circle_packing_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



_register("circle_packing_typed", "Circle packing: grid of disks with hashed radii (typed, node 273)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = v_uv * max(u_scale, 1.0);
    vec2 g = floor(uv);
    vec2 f = fract(uv) - 0.5;
    float h = hash21(g + 3.1);
    float rad = u_min_r + h * (u_max_r - u_min_r);
    float d = length(f);
    float disk = smoothstep(rad, rad - 0.05, d) * step(d, rad);
    float t = u_time * 0.05 * u_speed;
    vec3 tint = inferno(h);
    vec3 col = mix(u_bg, tint, disk);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":  {"glsl": "float", "min": 2.0, "max": 40.0, "default": 10.0,
               "description": "pack density"},
    "min_r":  {"glsl": "float", "min": 0.05, "max": 0.6, "default": 0.15,
               "description": "min disk radius"},
    "max_r":  {"glsl": "float", "min": 0.1, "max": 0.95, "default": 0.5,
               "description": "max disk radius"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
               "description": "animation speed"},
    "bg":     {"glsl": "color", "default": "#04060d", "description": "background"},
})