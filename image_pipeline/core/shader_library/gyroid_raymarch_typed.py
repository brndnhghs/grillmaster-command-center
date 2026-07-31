"""gyroid_raymarch_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Node 323: Raymarched 3D Gyroid TPMS — sphere-traced triply-periodic
#    minimal surface with lambert+specular lighting and camera orbit. Distinct
#    from node 301 (flat 2D slice of the scalar field): this sphere-traces the
#    thick-walled gyroid volume in 3D, so it reads as a solid woven lattice with
#    real depth, self-occlusion and shading. TPMS ref: Inigo Quilez raymarching
#    + gyroid f(p)=dot(sin(p),cos(p.yzx)).
_register("gyroid_raymarch_typed",
          "Raymarched 3D Gyroid TPMS — sphere-traced minimal surface (typed, node 323)",
          "procedural", '''float gyroid(vec3 p) {
    // Signed thick-shell distance to the gyroid iso-surface. The raw field is
    // dot(sin(p), cos(p.yzx)); subtracting u_bias thickens the walls (a
    // level-set band around 0), and the /scale factor is a Lipschitz correction
    // so sphere-tracing stays conservative.
    float f = dot(sin(p), cos(p.yzx));
    return (abs(f) - u_bias) / (u_freq * 2.5);
}
vec3 gyro_normal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        gyroid(p + e.xyy) - gyroid(p - e.xyy),
        gyroid(p + e.yxy) - gyroid(p - e.yxy),
        gyroid(p + e.yyx) - gyroid(p - e.yyx)));
}
void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * u_anim_speed;

    // Orbiting camera around the lattice.
    float ca = t * 0.3 + u_cam_angle * 6.2831853;
    vec3 ro = vec3(sin(ca), 0.35, cos(ca)) * u_cam_dist;
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw));
    vec3 up = cross(fw, rt);
    vec3 rd = normalize(fw + uv.x * rt * 1.2 + uv.y * up * 1.2);

    // Frequency scales the lattice; drift advances the field through space.
    vec3 drift = vec3(0.0, t * 0.4, 0.0);

    float d = 0.0;
    float hit = 0.0;
    vec3 pos = ro;
    for (int i = 0; i < 90; i++) {
        pos = ro + rd * d;
        float ds = gyroid(pos * u_freq + drift);
        if (ds < 0.0008) { hit = 1.0; break; }
        d += ds;
        if (d > 14.0) break;
    }

    vec3 col = u_bg;
    if (hit > 0.5) {
        vec3 n = gyro_normal(pos * u_freq + drift);
        vec3 ld = normalize(vec3(0.6, 0.8, 0.4));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        float spe = pow(clamp(dot(reflect(-ld, n), -rd), 0.0, 1.0), 32.0);
        // Depth-tinted base colour cycling with camera distance travelled.
        float depth = clamp(d / 10.0, 0.0, 1.0);
        vec3 base = mix(u_near_color, u_far_color, depth);
        col = base * (0.18 + 0.82 * dif) + vec3(1.0) * spe * u_spec;
        // Fake AO from march-step count folded into depth darkening.
        col *= 1.0 - depth * 0.35;
    }
    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));  // gamma
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "freq":       {"glsl": "float", "min": 0.4, "max": 4.0, "default": 1.4,
                   "description": "lattice frequency (cells per unit)"},
    "bias":       {"glsl": "float", "min": 0.0, "max": 1.2, "default": 0.35,
                   "description": "wall thickness (level-set band)"},
    "cam_dist":   {"glsl": "float", "min": 2.0, "max": 8.0, "default": 4.5,
                   "description": "camera distance from lattice"},
    "cam_angle":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
                   "description": "static camera azimuth offset (turns)"},
    "anim_speed": {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                   "description": "orbit + drift speed"},
    "spec":       {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.6,
                   "description": "specular highlight strength"},
    "near_color": {"glsl": "color", "default": "#ff7a3c",
                   "description": "near-surface colour"},
    "far_color":  {"glsl": "color", "default": "#2a6cff",
                   "description": "far-surface colour"},
    "bg":         {"glsl": "color", "default": "#05060a",
                   "description": "background colour"},
})