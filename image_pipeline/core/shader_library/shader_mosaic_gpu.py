"""shader_mosaic_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_mosaic_gpu", "GPU stained glass mosaic", "filter", _filter_typed('''
    float ts = u_tile_size;
    vec2 cell_uv = floor(uv * u_resolution / ts) * ts / u_resolution + ts / u_resolution * 0.5;
    f_color = texture(u_texture, cell_uv);
'''), uniforms={
    "tile_size": {"glsl": "float", "min": 10.0, "max": 60.0, "default": 30.0, "description": "tile size"},
})