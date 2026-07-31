"""god_rays_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



_register("god_rays_gpu", "God Rays (client-GPU twin of node 446)", "filter", _filter_typed('''
    vec2 light = vec2(u_light_x, 1.0 - u_light_y);
    vec2 delta = (uv - light) / 64.0 * u_density;
    vec2 pos = uv;
    float illum = 1.0;
    vec3 rays = vec3(0.0);
    float wsum = 0.0;                       // total weight for normalisation
    for (int i = 0; i < 64; i++) {
        pos -= delta;
        rays += texture(u_texture, pos).rgb * illum;
        wsum += illum;
        illum *= u_decay;
    }
    rays /= max(wsum, 1e-3);                // weighted average in [0,1]
    rays *= u_weight * u_exposure;          // gain controls
    float sr = max(u_sun_radius, 1.0) / max(u_resolution.x, u_resolution.y);
    float sun = exp(-dot(uv - light, uv - light) / (2.0 * sr * sr)) * u_sun_intensity;
    f_color = vec4(orig.rgb + rays * u_intensity + vec3(sun), 1.0);
'''), uniforms={
    "light_x": {"glsl": "float", "min": -0.5, "max": 1.5, "default": 0.30, "description": "light X position (normalised)"},
    "light_y": {"glsl": "float", "min": -0.5, "max": 1.5, "default": 0.28, "description": "light Y position (normalised, y-down)"},
    "density": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.92, "description": "ray length / distortion"},
    "decay": {"glsl": "float", "min": 0.80, "max": 0.99, "default": 0.95, "description": "per-sample illumination decay"},
    "weight": {"glsl": "float", "min": 0.1, "max": 1.0, "default": 0.5, "description": "per-sample contribution weight"},
    "exposure": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.6, "description": "final ray exposure"},
    "intensity": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "overall additive strength"},
    "sun_radius": {"glsl": "float", "min": 0.0, "max": 120.0, "default": 36.0, "description": "injected sun disc radius (px)"},
    "sun_intensity": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.6, "description": "injected sun brightness"},
})