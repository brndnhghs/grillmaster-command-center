"""marbling_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



_register("marbling_gpu", "Mathematical Marbling — closed-form fluid advection (client-GPU twin of node 953)", "procedural", _INFERNO + """
void main() {
    vec2 uv = v_uv;
    vec2 res = u_resolution;
    float t = u_time * u_anim_speed;
    float minwh = min(res.x, res.y);
    float base_r = u_drop_radius * minwh;
    float tstr = u_tine_strength * minwh;
    float tc = max(1e-3, u_tine_sharpness * minwh);
    vec3 bg = vec3(0.96);
    vec2 q = uv * res;
    for (int k = 0; k < 3; k++) {
        float fk = float(k);
        float ang = 6.2831853 * fk / 3.0 + 0.3;
        vec2 that = vec2(cos(ang), sin(ang));
        vec2 nhat = vec2(-that.y, that.x);
        float sweep = sin(t*0.6 + fk*1.3) * 0.5 + 0.5;
        vec2 p0 = vec2(sweep * res.x, (0.5 + 0.3*cos(fk)) * res.y);
        vec2 rel = q - p0;
        float along = dot(rel, that);
        float dd = abs(along);
        float decay = exp(-dd / tc);
        float disp = tstr * decay;
        q -= disp * nhat;
    }
    vec3 outc = bg;
    for (int i = 31; i >= 0; i--) {
        float fi = float(i);
        vec2 hc = vec2(hash21(vec2(fi + u_seed, 2.0)), hash21(vec2(fi + u_seed, 8.0)));
        vec2 ctr = (0.08 + 0.84*hc) * res;
        float rr = base_r * (0.6 + 0.8*hash21(vec2(fi + u_seed, 17.0)));
        vec2 d = q - ctr;
        if (dot(d,d) <= rr*rr) {
            float hh = hash21(vec2(fi + u_seed, 23.0));
            outc = 0.5 + 0.5*vec3(sin(hh*6.2831853), sin(hh*6.2831853+2.0943951), sin(hh*6.2831853+4.1887902));
        }
    }
    f_color = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
""",
    uniforms={
        "drop_radius": {"glsl": "float", "min": 0.01, "max": 0.3, "default": 0.09, "description": "base drop radius (fraction of min(W,H))"},
        "tine_strength": {"glsl": "float", "min": 0.0, "max": 0.6, "default": 0.22, "description": "tine displacement magnitude"},
        "tine_sharpness": {"glsl": "float", "min": 0.02, "max": 0.5, "default": 0.14, "description": "tine sharpness c (smaller = sharper)"},
        "anim_speed": {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed multiplier"},
        "seed": {"glsl": "float", "min": 0.0, "max": 99999.0, "default": 42.0, "description": "random seed for drop placement"},
    }
    )