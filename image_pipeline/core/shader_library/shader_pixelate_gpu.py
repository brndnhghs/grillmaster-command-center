"""shader_pixelate_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_pixelate_gpu", "GPU pixelation with edge preservation", "filter", _filter_typed('''
    float block = u_block;
    vec2 q = floor(uv * u_resolution / block) * block / u_resolution;
    f_color = texture(u_texture, q);
'''), uniforms={
    "block": {"glsl": "float", "min": 4.0, "max": 64.0, "default": 16.0, "description": "pixel block size"},
})