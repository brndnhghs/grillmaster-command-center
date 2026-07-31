"""shader_emboss — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("shader_emboss", "GPU emboss / bump mapping", "filter", _filter_typed('''
    float gx = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float gy = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    float gz = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float dx = gx - gz; float dy = gy - gz;
    float bump = (dx + dy) * 2.0 + 0.5;
    f_color = vec4(mix(orig.rgb, vec3(bump), u_strength), 1.0);
'''), uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.8, "description": "emboss blend"},
})