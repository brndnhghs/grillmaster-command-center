"""shader_caustics_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_caustics_gpu", "GPU caustic light overlay", "filter", _filter_typed('''
    float caustic = sin(uv.x * 30.0 + u_time) * cos(uv.y * 25.0 + u_time * 0.7);
    caustic += sin(uv.x * 50.0 - u_time * 1.3) * sin(uv.y * 40.0 + u_time * 0.5) * 0.5;
    caustic = max(0.0, caustic) * u_amount;
    vec3 light = vec3(0.8, 0.9, 1.0) * caustic;
    f_color = vec4(orig.rgb + light, 1.0);
'''), uniforms={
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "caustic strength"},
})