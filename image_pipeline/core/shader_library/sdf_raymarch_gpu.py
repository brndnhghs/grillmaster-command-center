"""sdf_raymarch_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── FILTER SHADERS (process input image) ──

_register("sdf_raymarch_gpu", "SDF raymarching (signed-distance-field scene)", "procedural", '''
// GPU twin of CPU node 412 — real-time ray marching of a smooth-min
// union of signed-distance primitives (a la Inigo Quilez / Shadertoy).
//   u_params.x = camera distance      (0.5 -> 5.0)
//   u_params.y = scene complexity, # primitives (0.5 -> 4)
//   u_params.z = march steps        (0.5 -> 90)
//   u_params.w = rim glow amount     (0.5 -> 0.5)
// u_time rotates the camera + spins the primitives.
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float mapScene(vec3 p) {
    float nSph = floor(u_primitives + 0.5);
    float k = 0.4;
    float d = 1e5;
    for (int i = 0; i < 6; i++) {
        if (float(i) >= nSph) break;
        float a = float(i) / max(nSph, 1.0) * 6.2831853 + u_time * 0.3;
        vec3 c = vec3(cos(a) * 1.6, sin(float(i) * 1.3) * 0.4, sin(a) * 1.6);
        float ds = length(p - c) - 0.7;
        d = smin(d, ds, k);
    }
    float ground = p.y + 1.2;
    d = smin(d, ground, 0.5);
    return d;
}
vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.0015, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy) - mapScene(p - e.xyy),
        mapScene(p + e.yxy) - mapScene(p - e.yxy),
        mapScene(p + e.yyx) - mapScene(p - e.yyx)));
}
void main() {
    float camDist = u_cam_dist;
    float glow = u_glow;
    float steps = floor(u_steps + 0.5);

    vec2 uv = (v_uv * 2.0 - 1.0);
    uv.x *= u_resolution.x / u_resolution.y;

    float ang = u_time * 0.5;
    vec3 ro = vec3(sin(ang) * camDist, 0.6, cos(ang) * camDist);
    vec3 ta = vec3(0.0);
    vec3 fwd = normalize(ta - ro);
    vec3 rgt = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 upv = cross(fwd, rgt);
    vec3 rd = normalize(fwd * 1.5 + uv.x * rgt + uv.y * upv);

    float t = 0.0;
    float hit = 0.0;
    vec3 p = ro;
    for (int i = 0; i < 160; i++) {
        if (float(i) >= steps) break;
        p = ro + rd * t;
        float d = mapScene(p);
        if (d < 0.001) { hit = 1.0; break; }
        t += d;
        if (t > 30.0) break;
    }

    vec3 bg = mix(vec3(0.05, 0.07, 0.12), vec3(0.18, 0.22, 0.34), uv.y * 0.5 + 0.5);
    vec3 col = bg;
    if (hit > 0.5) {
        vec3 n = calcNormal(p);
        vec3 lig = normalize(vec3(0.6, 0.7, -0.4));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        float amb = 0.3 + 0.7 * n.y;
        vec3 base = u_surface;
        col = base * (amb * 0.4 + dif * 0.9);
        float rim = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 2.0);
        col += glow * rim * vec3(0.35, 0.7, 1.0);
    }
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cam_dist":   {"glsl": "float", "min": 3.0, "max": 7.0, "default": 5.0,
                   "description": "camera orbit distance"},
    "primitives": {"glsl": "float", "min": 2.0, "max": 6.0, "default": 4.0,
                   "description": "number of blended SDF spheres"},
    "steps":      {"glsl": "float", "min": 40.0, "max": 140.0, "default": 90.0,
                   "description": "max ray-march steps (quality)"},
    "glow":       {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "rim-glow amount"},
    "surface":    {"glsl": "color", "default": "#e68c59",
                   "description": "surface base color"},
})