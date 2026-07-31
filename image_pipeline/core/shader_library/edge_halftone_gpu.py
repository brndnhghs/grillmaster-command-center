"""edge_halftone_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# 64 Edge Halftone — Sobel-magnitude-weighted dots (GPU live twin)
_register("edge_halftone_gpu", "GPU edge-weighted halftone dots", "filter", _filter_typed('''
    float tl = dot(texture(u_texture, uv + vec2(-step.x, -step.y)).rgb, vec3(0.299,0.587,0.114));
    float t  = dot(texture(u_texture, uv + vec2(0, -step.y)).rgb, vec3(0.299,0.587,0.114));
    float tr = dot(texture(u_texture, uv + vec2(step.x, -step.y)).rgb, vec3(0.299,0.587,0.114));
    float l  = dot(texture(u_texture, uv + vec2(-step.x, 0)).rgb, vec3(0.299,0.587,0.114));
    float r  = dot(texture(u_texture, uv + vec2(step.x, 0)).rgb, vec3(0.299,0.587,0.114));
    float bl = dot(texture(u_texture, uv + vec2(-step.x, step.y)).rgb, vec3(0.299,0.587,0.114));
    float b  = dot(texture(u_texture, uv + vec2(0, step.y)).rgb, vec3(0.299,0.587,0.114));
    float br = dot(texture(u_texture, uv + vec2(step.x, step.y)).rgb, vec3(0.299,0.587,0.114));
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = clamp(sqrt(gx*gx + gy*gy), 0.0, 1.0);
    float cell = 4.0 + u_dot_spacing * 16.0;               // dot_spacing
    float base = (1.0 - edge) * 0.5 * (0.5 + u_dot_size * 0.5); // dot_size
    vec2 q = fract(uv * u_resolution / cell) - 0.5;
    float d = length(q);
    float dot_r = clamp(base, 0.02, 0.5);
    float v = d < dot_r ? 0.0 : 1.0;
    vec3 bg = vec3(0.05, 0.05, 0.08);
    f_color = vec4(mix(bg, vec3(1.0), v), 1.0);
'''), uniforms={
        "dot_spacing": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "halftone cell spacing"},
        "dot_size": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "halftone dot size"},
    })