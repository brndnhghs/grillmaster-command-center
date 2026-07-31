"""shader_glitch_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_glitch_gpu", "GPU digital glitch artifacts", "filter", _filter_typed('''
    float band = floor(uv.y * 40.0 * u_intensity);
    float shift = sin(band * 7.0 + u_time * 5.0) * 0.05 * u_intensity;
    float noise = fract(sin(dot(uv * u_resolution, vec2(12.9898, 78.233))) * 43758.5453);
    float glitch = noise > (1.0 - u_intensity * 0.1) ? 1.0 : 0.0;
    vec2 q = uv + vec2(shift + glitch * 0.1, 0.0);
    f_color = texture(u_texture, q);
'''), uniforms={
    "intensity": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "glitch intensity"},
})