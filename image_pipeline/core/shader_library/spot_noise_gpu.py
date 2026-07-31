"""spot_noise_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("spot_noise_gpu", "Spot Noise — flow-oriented anisotropic spots (client-GPU twin of node 534)", "procedural", _INFERNO + """
void main() {
    vec2 uv = v_uv;
    vec2 res = u_resolution;
    float t = u_time * u_anim_speed;
    float contrast = u_contrast;
    float spot = u_spot_size;
    float stretch = u_stretch;
    float field = 0.0;
    for (int i = 0; i < 64; i++) {
        float fi = float(i);
        vec2 hc = vec2(hash21(vec2(fi, 1.0)), hash21(vec2(fi, 7.0)));
        vec2 c = hc * res;
        vec2 rel = (c / res) - 0.5;
        float theta = atan(rel.y, rel.x) + 1.5707963;
        c += vec2(cos(theta), sin(theta)) * (t * 22.0);
        c = mod(c, res);
        theta += t * 0.4;
        float ct = cos(theta), st = sin(theta);
        float sa = spot * stretch;
        float sb = spot / sqrt(max(stretch, 1e-3));
        vec2 d = (uv * res) - c;
        float uu = ct*d.x + st*d.y;
        float vv = -st*d.x + ct*d.y;
        float g = exp(-(uu*uu/(2.0*sa*sa) + vv*vv/(2.0*sb*sb)));
        float amp = (hash21(vec2(fi, 13.0)) - 0.5) * 2.0;
        amp *= (0.5 + 0.5*sin(t*0.7));
        field += amp * g;
    }
    float val = 0.5 + field / (0.6 * 4.0);
    val = clamp(0.5 + (val - 0.5) * contrast, 0.0, 1.0);
    f_color = vec4(inferno(val), 1.0);
}
""",
    uniforms={
        "spot_size": {"glsl": "float", "min": 3.0, "max": 40.0, "default": 14.0, "description": "base spot radius in px (before stretch)"},
        "stretch": {"glsl": "float", "min": 1.0, "max": 12.0, "default": 5.0, "description": "anisotropy: elongation along the flow direction"},
        "contrast": {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.4, "description": "final tone contrast"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0, "description": "animation speed multiplier"},
    }
    )