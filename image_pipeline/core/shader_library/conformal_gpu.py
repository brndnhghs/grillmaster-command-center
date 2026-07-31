"""conformal_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Node 503: Conformal Warp — Möbius map deforms a base grid (client-GPU twin) ──
_register("conformal_gpu", "Conformal warp — Möbius map deforms a base grid (client-GPU twin of node 503)", "procedural", '''
vec2 cz_mul(vec2 a, vec2 b){ return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cz_conj(vec2 a){ return vec2(a.x, -a.y); }
vec2 cz_div(vec2 a, vec2 b){ float d = dot(b,b) + 1e-6; return vec2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / d; }
float cz_ss(float a, float b, float x){ float t = clamp((x-a)/(b-a), 0.0, 1.0); return t*t*(3.0-2.0*t); }

void main() {
    vec2 res = u_resolution;
    float aspect = res.x / res.y;
    // centered, aspect-correct complex coordinate
    vec2 z = (v_uv - 0.5) * 2.0;
    z.x *= aspect;
    float scale = clamp(u_scale * 6.0, 0.5, 8.0);   // domain radius
    z *= scale;
    float warp = clamp(u_warp, 0.0, 0.92);          // |a|
    float anim = u_time * clamp(u_anim_speed * 2.0, 0.05, 4.0); // anim speed

    // Möbius coefficient a orbits inside the unit disk => disk->disk conformal map
    vec2 a = warp * vec2(cos(anim), sin(anim));
    // also rotate the input plane for extra life
    float grot = anim * 0.5;
    z = vec2(z.x*cos(grot) - z.y*sin(grot), z.x*sin(grot) + z.y*cos(grot));
    vec2 num = z - a;
    vec2 den = vec2(1.0, 0.0) - cz_mul(cz_conj(a), z);
    vec2 w = cz_div(num, den);

    // base grid pattern sampled at the warped coordinate
    float k = 6.0;
    vec2 g = w * k;
    float fx = abs(fract(g.x) - 0.5);
    float fy = abs(fract(g.y) - 0.5);
    float line = cz_ss(0.40, 0.49, max(fx, fy));
    float r = length(w);
    float shade = clamp(line * (0.55 + 0.45 * sin(r * 2.5)), 0.0, 1.0);
    vec3 base = mix(vec3(0.04, 0.06, 0.11), vec3(0.92, 0.96, 1.0), shade);
    // subtle hue from argument(w)
    float ang = atan(w.y, w.x);
    vec3 tint = 0.14 * vec3(cos(ang), cos(ang + 2.094), cos(ang + 4.188)) * (1.0 - shade);
    f_color = vec4(base + tint, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "domain radius (scaled x6 internally, clamped 0.5-8.0)"},
    "warp":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "Mobius coefficient magnitude |a| (0-0.92)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "sun/orbit animation speed (scaled x2 internally)"},
})