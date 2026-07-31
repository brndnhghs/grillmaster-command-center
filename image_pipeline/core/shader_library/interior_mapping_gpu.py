"""interior_mapping_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── 967 Interior Mapping (client-GPU twin) ──
_register("interior_mapping_gpu",
          "Interior Mapping (client-GPU twin of node 967)",
          "procedural",
'''void main() {
    // Fake 3D rooms behind a flat facade (van Dongen 2008): per-pixel ray-box
    // intersection into a tiled window grid. Continuous camera pan keeps it live.
    vec2 gu = gl_FragCoord.xy / u_resolution;
    float t = u_time * u_anim_speed;

    float pan_x = u_pan_x + 0.6 * sin(t);
    float pan_y = u_pan_y + 0.25 * cos(t * 0.7);
    float persp = u_perspective;

    vec2 f = gu * vec2(u_n_cols, u_n_rows);
    vec2 ci = floor(f);
    vec2 lxy = (f - ci) - 0.5;                    // local window coord [-0.5,0.5]
    float lx = lxy.x, ly = lxy.y;

    bool in_frame = (abs(lx) > (0.5 - u_frame_width)) || (abs(ly) > (0.5 - u_frame_width));

    // per-window hashed room params
    float rd = hash21(ci + 1.0);
    float depth = u_room_depth * (0.75 + 0.5 * rd);
    float wall_h = hash21(ci + 2.0);
    float lit_h = hash21(ci + 3.0);
    // flicker lit set over time
    float flick = 0.5 + 0.5 * sin(t * (0.6 + lit_h * 2.0) + wall_h * 6.2831853);
    float lit_val = clamp(lit_h * 0.5 + flick * 0.5, 0.0, 1.0);
    bool lit = lit_val < u_lit_fraction;

    // cast interior ray, intersect room box
    float dz = 1.0;
    float dx = lx * persp + pan_x;
    float dy = ly * persp + pan_y;
    float eps = 1e-6;
    float dxs = abs(dx) < eps ? eps : dx;
    float dys = abs(dy) < eps ? eps : dy;
    float tz = depth / dz;
    float tx = dx > 0.0 ? (0.5 - lx) / dxs : (-0.5 - lx) / dxs;
    float ty = dy > 0.0 ? (0.5 - ly) / dys : (-0.5 - ly) / dys;
    tx = tx > 0.0 ? tx : 1e9;
    ty = ty > 0.0 ? ty : 1e9;
    float t_hit = min(min(tz, tx), ty);
    bool hit_back = (tz <= tx) && (tz <= ty);
    bool hit_side = (tx < tz) && (tx <= ty);
    bool hit_ud = (ty < tz) && (ty < tx);

    float hx = lx + dx * t_hit;
    float hy = ly + dy * t_hit;
    float hz = dz * t_hit;

    // base wall colour warm vs cool
    vec3 warm = vec3(0.62, 0.50, 0.38);
    vec3 cool = vec3(0.40, 0.45, 0.52);
    vec3 base = warm * u_warmth + cool * (1.0 - u_warmth);
    vec3 rgb = base * (0.75 + 0.5 * wall_h);

    float shade = 1.0;
    if (hit_side) shade = 0.82;
    if (hit_ud && hy > 0.0) shade = 1.05;
    if (hit_ud && hy <= 0.0) shade = 0.62;
    rgb *= shade;

    // back-wall picture detail
    float bx = hx + 0.5, by = hy + 0.5;
    if (hit_back && abs(bx - 0.5) < 0.22 && abs(by - 0.55) < 0.16)
        rgb = vec3(0.20, 0.28, 0.42);

    // depth attenuation
    float dnorm = clamp(hz / (u_room_depth * 1.3), 0.0, 1.0);
    rgb *= (1.0 - 0.55 * dnorm);

    // ceiling light glow for lit rooms
    float gl = exp(-((hx * hx) / 0.12 + ((hy - 0.35) * (hy - 0.35)) / 0.10));
    if (lit) rgb += vec3(1.0, 0.92, 0.72) * (gl * 0.9);
    rgb *= lit ? 1.0 : 0.30;
    if (!lit) rgb *= vec3(0.7, 0.8, 1.0);

    // faint sky reflection
    rgb += vec3(0.10, 0.14, 0.22) * (1.0 - gu.y) * 0.12;

    // facade mullions
    if (in_frame) rgb = vec3(0.14, 0.14, 0.16);

    f_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
''',
uniforms={
    "n_cols":      {"glsl": "float", "min": 1.0, "max": 24.0, "default": 8.0, "description": "window columns across facade"},
    "n_rows":      {"glsl": "float", "min": 1.0, "max": 24.0, "default": 6.0, "description": "window rows down facade"},
    "room_depth":  {"glsl": "float", "min": 0.4, "max": 3.0, "default": 1.4, "description": "virtual room depth"},
    "perspective": {"glsl": "float", "min": 0.2, "max": 2.5, "default": 1.1, "description": "parallax strength"},
    "pan_x":       {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "horizontal camera offset"},
    "pan_y":       {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.15, "description": "vertical camera offset"},
    "frame_width": {"glsl": "float", "min": 0.0, "max": 0.25, "default": 0.06, "description": "facade mullion thickness"},
    "lit_fraction":{"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6, "description": "fraction of lit windows"},
    "warmth":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "room colour warmth"},
    "anim_speed":  {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
})