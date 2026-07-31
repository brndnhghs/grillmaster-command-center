"""mandelbulb_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Node 331: Mandelbulb — 3D escape-time fractal (Daniel White & Paul
# Nylander 2009), sphere-traced with the Hart, Sandin & Kauffman (1989)
# analytical distance estimator. Distinct from the 2D escape-time family
# (Mandelbrot/Julia/Burning Ship) and from the 3D TPMS raymarches (gyroid/
# menger): this is the canonical "3D Mandelbrot". The power exponent slowly
# morphs with u_time (sin breathing) so the bulb is genuinely time-varying
# (avoids the contrast-only static cull), and the camera orbits it. ──
_register("mandelbulb_gpu",
          "Mandelbulb — 3D escape-time fractal (White & Nylander 2009), sphere-traced "
          "via the Hart et al. 1989 distance estimator; animated power morph + orbiting camera",
          "procedural", '''float mandelDE(vec3 pos, float power, out float orbit) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    float trap = 0.0;
    for (int i = 0; i < 24; i++) {
        if (float(i) >= u_iterations) break;
        r = length(z);
        if (r > u_bailout) break;
        float theta = acos(clamp(z.z / max(r, 1e-6), -1.0, 1.0));
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        float zr = pow(r, power);
        theta *= power;
        phi *= power;
        z = zr * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + pos;
        trap = float(i);
    }
    orbit = clamp(trap / max(u_iterations, 1.0), 0.0, 1.0);
    return 0.5 * log(max(r, 1.0001)) * r / max(dr, 1e-6);
}
vec3 mandelNormal(vec3 p, float power) {
    vec2 e = vec2(0.0008, 0.0);
    float oa, ob, oc, od;
    return normalize(vec3(
        mandelDE(p + e.xyy, power, oa) - mandelDE(p - e.xyy, power, ob),
        mandelDE(p + e.yxy, power, oc) - mandelDE(p - e.yxy, power, od),
        mandelDE(p + e.yyx, power, oa) - mandelDE(p - e.yyx, power, ob)));
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_anim_speed;
    // Power morph: the signature bulb shape breathes between powers so the
    // fractal genuinely animates even with a static camera.
    float power = clamp(u_power + 1.0 * sin(t * 0.3), 2.0, 12.0);

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
        float ds = mandelDE(pos, power, o);
        orbit = o;
        if (ds < 0.0008) { hit = 1.0; break; }
        d += ds;
        steps = i;
        if (d > 12.0) break;
    }

    vec3 col = u_bg;
    if (hit > 0.5) {
        vec3 n = mandelNormal(pos, power);
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
    "power":      {"glsl": "float", "min": 2.0, "max": 12.0, "default": 8.0,
                   "description": "fractal exponent (signature bulb power)"},
    "iterations": {"glsl": "float", "min": 4.0, "max": 20.0, "default": 12.0,
                   "description": "escape-time iteration cap (detail)"},
    "cam_dist":   {"glsl": "float", "min": 1.5, "max": 5.0, "default": 2.6,
                   "description": "camera distance from the bulb"},
    "cam_angle":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "static camera azimuth offset (turns)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "camera orbit + power-morph speed"},
    "bailout":    {"glsl": "float", "min": 2.0, "max": 8.0, "default": 4.0,
                   "description": "escape radius (shape / level of detail)"},
    "spec":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "specular highlight strength"},
    "base_color": {"glsl": "color", "default": "#ffb24a",
                   "description": "surface base colour"},
    "glow_color": {"glsl": "color", "default": "#3aa0ff",
                   "description": "orbit-trap glow colour"},
    "bg":         {"glsl": "color", "default": "#05060a",
                   "description": "background colour"},
})