"""matcap_relight_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── 923 MatCap Relight (2.5D normal-from-luminance shading) ──
_register("matcap_relight_gpu", "MatCap Relight (client-GPU twin of node 923)", "filter", _filter_typed('''
    float l0 = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float lx = dot(texture(u_texture, uv + vec2(step.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float ly = dot(texture(u_texture, uv + vec2(0.0, step.y)).rgb, vec3(0.299, 0.587, 0.114));
    vec3 n = normalize(vec3(-(lx - l0) * u_relief,
                            -(ly - l0) * u_relief, 1.0));
    vec3 L = normalize(vec3(cos(u_light_dir), sin(u_light_dir), 0.7));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 halfv = normalize(L + V);
    float ndl = clamp(dot(n, L), 0.0, 1.0);
    float ndh = clamp(dot(n, halfv), 0.0, 1.0);
    float spec = pow(ndh, u_spec_pow);
    float fres = pow(clamp(1.0 - n.z, 0.0, 1.0), 3.0);
    vec3 base = vec3(0.85, 0.55, 0.35);
    vec3 outc = base * (0.25 + 0.75 * ndl)
              + 0.12 * fres * vec3(1.0, 0.9, 0.8)
              + spec * vec3(1.0);
    outc = outc * u_strength + (1.0 - u_strength) * 0.5;
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
'''), uniforms={
    "light_dir": {"glsl": "float", "min": 0.0, "max": 6.2832, "default": 0.6, "description": "key light azimuth (rad)"},
    "relief": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0, "description": "surface relief / depth gain"},
    "strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "mix toward matcap (0=flat, 1=full)"},
    "spec_pow": {"glsl": "float", "min": 2.0, "max": 128.0, "default": 24.0, "description": "specular exponent"},
})