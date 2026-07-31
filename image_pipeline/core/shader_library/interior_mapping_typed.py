"""interior_mapping_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




_register("interior_mapping_typed",
          "Interior Mapping — fake 3D rooms behind a facade via ray-box intersection (van Dongen 2008, typed, node 328)",
          "procedural", '''
// Interior Mapping (Joost van Dongen, CGI 2008): render believable interior
// rooms behind a flat facade WITHOUT any extra geometry. For each screen pixel
// we build an eye ray into a virtual room-box (one cell of a repeating grid of
// rooms) and intersect it against the 6 walls; the nearest wall behind the
// facade plane is shaded with a simple depth-tinted colour + a glowing window
// light. Ceiling/floor/back/side walls each get their own tint so the illusion
// of depth reads clearly. Windows are lit per-room by a hashed on/off so the
// building looks inhabited, and u_time drives a slow twinkle.
float _im_hash(vec2 c){ return fract(sin(dot(c, vec2(41.3, 289.7))) * 43758.5453); }
void main() {
    vec2 uv = v_uv;
    float aspect = u_resolution.x / u_resolution.y;
    // Facade plane coords, scaled into a grid of rooms.
    vec2 fp = vec2(uv.x * aspect, uv.y) * u_rooms;
    vec2 cell = floor(fp);            // which room
    vec2 f = fract(fp);               // position on the facade within the room [0,1]

    // Per-room hash → window-lit state + slight room-depth variation.
    float h = _im_hash(cell);
    float lit = step(1.0 - u_lit_frac, _im_hash(cell + 3.1));
    // Twinkle: lit rooms flicker slowly.
    float tw = 0.75 + 0.25 * sin(u_time * u_speed * 2.0 + h * 30.0);
    lit *= tw;

    // Eye ray. Camera sits in front of the facade (z = -dist); we look toward
    // +z into the room. Screen offset from room centre gives ray direction.
    float depth = u_depth * (0.7 + 0.6 * h);   // room depth varies per room
    vec3 ro = vec3(0.0, 0.0, -u_cam_dist);
    // Ray direction: parallax from facade-local coords centred at 0.
    vec3 rd = normalize(vec3((f - 0.5) * u_parallax, 1.0));

    // Room box spans x,y in [-0.5,0.5], z in [0, depth]. Intersect ray with the
    // far planes; pick the nearest positive-z surface = the wall we see.
    // Slab method on each axis for the exit face.
    vec3 inv = 1.0 / rd;
    // x walls at +/-0.5, y walls at +/-0.5, back wall at z=depth.
    float tx = ((rd.x > 0.0 ? 0.5 : -0.5) - ro.x) * inv.x;
    float ty = ((rd.y > 0.0 ? 0.5 : -0.5) - ro.y) * inv.y;
    float tz = (depth - ro.z) * inv.z;
    float tHit = min(min(tx, ty), tz);
    vec3 hit = ro + rd * tHit;

    // Classify which wall was hit → base tint.
    vec3 wallCol;
    float shade;
    if (tHit == tz) {
        wallCol = u_back;                       // back wall
        shade = 1.0;
    } else if (tHit == ty) {
        wallCol = (rd.y > 0.0) ? u_ceiling : u_floor;
        shade = 0.85;
    } else {
        wallCol = u_side;                        // side walls
        shade = 0.7;
    }
    // Depth fog: deeper points darker.
    float zf = clamp(hit.z / max(depth, 0.001), 0.0, 1.0);
    shade *= mix(1.0, u_ambient, zf);

    // Window glow: a warm rectangle on the back wall for lit rooms.
    vec3 col = wallCol * shade;
    if (lit > 0.001) {
        // back-wall panel glow, brighter toward the back.
        float panel = smoothstep(0.55, 0.35, abs(hit.x)) *
                      smoothstep(0.55, 0.35, abs(hit.y));
        col += u_light * lit * panel * (0.4 + 0.6 * zf);
    }

    // Thin mullion frame on the facade between rooms (window bars).
    float bar = min(smoothstep(0.0, u_frame, f.x) * smoothstep(1.0, 1.0 - u_frame, f.x),
                    smoothstep(0.0, u_frame, f.y) * smoothstep(1.0, 1.0 - u_frame, f.y));
    col = mix(u_frame_col, col, bar);

    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "rooms":     {"glsl": "float", "min": 1.0, "max": 20.0, "default": 7.0,
                  "description": "rooms across the facade (grid density)"},
    "depth":     {"glsl": "float", "min": 0.3, "max": 4.0, "default": 1.5,
                  "description": "room depth (parallax strength)"},
    "cam_dist":  {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                  "description": "camera distance from facade"},
    "parallax":  {"glsl": "float", "min": 0.2, "max": 2.0, "default": 1.0,
                  "description": "eye-ray spread (viewing angle)"},
    "lit_frac":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5,
                  "description": "fraction of rooms with lights on"},
    "speed":     {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                  "description": "window twinkle speed"},
    "ambient":   {"glsl": "float", "min": 0.05, "max": 1.0, "default": 0.3,
                  "description": "deep-room ambient floor"},
    "frame":     {"glsl": "float", "min": 0.0, "max": 0.3, "default": 0.06,
                  "description": "window mullion frame width"},
    "back":      {"glsl": "color", "default": "#3a4152", "description": "back wall colour"},
    "ceiling":   {"glsl": "color", "default": "#5a6273", "description": "ceiling colour"},
    "floor":     {"glsl": "color", "default": "#2a2e38", "description": "floor colour"},
    "side":      {"glsl": "color", "default": "#454b5a", "description": "side wall colour"},
    "light":     {"glsl": "color", "default": "#ffd28a", "description": "window light colour"},
    "frame_col": {"glsl": "color", "default": "#12141a", "description": "mullion frame colour"},
})