"""mandelbox_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Node 336: Mandelbox — 3D escape-time fractal (Tom Lowe, 2010), the
# box-fold + sphere-fold companion to the Mandelbulb. Each iteration applies a
# per-axis mirror "box fold" (z -> 2*clamp(z,-s,s) - z) then a radius-clamped
# "sphere fold" (pull near-origin points outward, push far points in) and the
# scale*+c affine map; the Hart et al. 1989 scalar distance estimator divides
# the length of z by |dr| (the accumulated derivative) to give a raymarchable
# DE. The negative scale (default -1.5) yields the iconic tiled infinite-rooms
# look, distinct from the bulb (power morph) and from the KIFS wedge-fold
# (node 402/330). An orbiting camera + a subtle scale breathing keep it
# genuinely time-varying so animation drivers have a visibly-responsive target
# and the contrast-only static liveness cull is avoided. ──
_register("mandelbox_gpu",
          "Mandelbox — 3D escape-time fractal (Tom Lowe 2010), box-fold + sphere-fold "
          "DE raymarch (Hart et al. 1989); orbiting camera + scale breathing",
          "procedural", '''float mandelboxDE(vec3 p, float scale, out float orbit) {
    vec3 z = p;
    vec3 c = p;            // Mandelbox: offset c = starting point (connected tiling)
    float dr = 1.0;
    float trap = 0.0;
    float fr2 = u_fixed_radius * u_fixed_radius;
    float mr2 = u_min_radius * u_min_radius;
    for (int i = 0; i < 24; i++) {
        if (float(i) >= u_iterations) break;
        // Box fold: mirror z back into the [-fold, fold] box.
        z = clamp(z, -u_fold, u_fold) * 2.0 - z;
        // Sphere fold: conditional radius rescale.
        float r2 = dot(z, z);
        if (r2 < mr2) {
            float t = fr2 / mr2;
            z *= t; dr *= t;
        } else if (r2 < fr2) {
            float t = fr2 / max(r2, 1e-6);
            z *= t; dr *= t;
        }
        z = scale * z + c;
        dr = dr * abs(scale) + 1.0;
        trap = float(i);
        if (dot(z, z) > 16.0) break;   // escape
    }
    orbit = clamp(trap / max(u_iterations, 1.0), 0.0, 1.0);
    return length(z) / abs(dr);
}
vec3 mandelboxNormal(vec3 p, float scale) {
    vec2 e = vec2(0.0009, 0.0);
    float oa, ob, oc, od;
    return normalize(vec3(
        mandelboxDE(p + e.xyy, scale, oa) - mandelboxDE(p - e.xyy, scale, ob),
        mandelboxDE(p + e.yxy, scale, oc) - mandelboxDE(p - e.yxy, scale, od),
        mandelboxDE(p + e.yyx, scale, oa) - mandelboxDE(p - e.yyx, scale, ob)));
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_anim_speed;
    // Scale breathing: keep the iconic negative-scale tiling while morphing.
    float scale = clamp(u_scale + 0.25 * sin(t * 0.25), -2.5, 3.0);

    float ca = t * 0.3 + u_cam_angle * 6.2831853;
    vec3 ro = vec3(sin(ca), 0.35, cos(ca)) * u_cam_dist;
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 rd = normalize(fw + uv.x * rt * 1.3 + uv.y * up * 1.3);

    float d = 0.0;
    float hit = 0.0;
    vec3 pos = ro;
    float orbit = 0.0;
    int steps = 0;
    for (int i = 0; i < 160; i++) {
        pos = ro + rd * d;
        float o;
        float ds = mandelboxDE(pos, scale, o);
        orbit = o;
        if (ds < 0.0008) { hit = 1.0; break; }
        d += ds;
        steps = i;
        if (d > 12.0) break;
    }

    vec3 col = u_bg;
    if (hit > 0.5) {
        vec3 n = mandelboxNormal(pos, scale);
        vec3 ld = normalize(vec3(0.6, 0.8, 0.4));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        float spe = pow(clamp(dot(reflect(-ld, n), -rd), 0.0, 1.0), 24.0);
        vec3 base = mix(u_base_color, u_glow_color, orbit);
        col = base * (0.2 + 0.8 * dif) + vec3(1.0) * spe * u_spec;
        col += u_glow_color * (1.0 - orbit) * 0.4;  // inner-trap glow
        float ao = 1.0 - float(steps) / 160.0 * 0.4;  // step-count fake AO
        col *= ao;
    }
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": -2.5, "max": 3.0, "default": -1.5,
                   "description": "fractal scale (negative = iconic tiled-rooms look)"},
    "fold":       {"glsl": "float", "min": 0.1, "max": 2.0, "default": 1.0,
                   "description": "box-fold clamp half-width"},
    "min_radius": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.5,
                   "description": "sphere-fold inner radius"},
    "fixed_radius": {"glsl": "float", "min": 0.3, "max": 2.0, "default": 1.0,
                     "description": "sphere-fold outer radius"},
    "iterations": {"glsl": "float", "min": 4.0, "max": 24.0, "default": 14.0,
                   "description": "escape-time iteration cap (detail)"},
    "cam_dist":   {"glsl": "float", "min": 2.0, "max": 12.0, "default": 6.0,
                   "description": "camera distance from the box"},
    "cam_angle":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "static camera azimuth offset (turns)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "camera orbit + scale-breathing speed"},
    "spec":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "specular highlight strength"},
    "base_color": {"glsl": "color", "default": "#e8c07a",
                   "description": "surface base colour"},
    "glow_color": {"glsl": "color", "default": "#3aa0ff",
                   "description": "orbit-trap glow colour"},
    "bg":         {"glsl": "color", "default": "#05060a",
                   "description": "background colour"},
})