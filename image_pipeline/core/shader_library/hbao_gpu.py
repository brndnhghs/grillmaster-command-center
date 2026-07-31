"""hbao_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _HBAO_HELPERS



_register("hbao_gpu",
          "Horizon-Based Ambient Occlusion of a procedural height field (node 425)",
          "procedural",
          _HBAO_HELPERS + '''
void main() {
    // ── Named uniforms (auto-declared) match node 425's REAL params ──
    // freq, octaves, height_scale, radius, directions, steps, jitter,
    // mode (choice), light_az, light_el, ambient, contrast, colormode (choice).
    // u_time drives the anim_mode (evolve/drift/rotate_light) on the GPU path.
    vec2 res = u_resolution;
    vec2 uv = v_uv;
    vec2 p = (uv - 0.5) * 2.0;

    // ── Animation (GPU live-preview clock) ──
    int amode = int(clamp(floor(u_anim_mode + 0.5), 0.0, 3.0)); // 0 none,1 evolve,2 drift,3 rotate_light
    float t = u_time;
    vec2 off = vec2(0.0);
    float wblend = 0.0;
    if (amode == 2) { off = vec2(t * 0.18, t * 0.08); }       // drift
    else if (amode == 1) { wblend = 0.5 - 0.5 * cos(t); }     // evolve (full 0..1 blend)

    // ── Procedural height field ──
    vec2 fcoord = p * max(u_freq, 0.5);
    float h = _hbao_fbm(fcoord + off, int(clamp(u_octaves, 1.0, 6.0)));
    if (amode == 1) {
        float ang = t * 0.6;
        float ca = cos(ang), sa = sin(ang);
        vec2 rx = fcoord * ca - fcoord.y * sa;
        vec2 ry = fcoord * sa + fcoord.y * ca;
        float h2 = _hbao_fbm(rx + off, int(clamp(u_octaves, 1.0, 6.0)));
        h = mix(h, h2, wblend);
    }
    h = h * 0.5 + 0.5;  // [-1,1] -> [0,1]

    // ── Per-pixel disk rotation (jitter) ──
    float jr = fract(_hbao_hash(uv * res * 0.013 + 3.1)
                     + _hbao_hash(uv * res * 0.027 + 7.7));
    float rot = jr * 6.2831853 * u_jitter;
    float cr = cos(rot), sr = sin(rot);

    int D = int(clamp(u_directions, 3.0, 16.0));
    int K = int(clamp(u_steps, 4.0, 32.0));
    float step_px = max(1.0, u_radius) / float(K);
    // World-height slope factor (EXAG keeps smooth-field relief non-trivial).
    float k = u_height_scale * 12.0;

    float vis_sum = 0.0;
    for (int d = 0; d < 16; d++) {
        if (d >= D) break;
        float base_ang = (6.2831853 / float(D)) * float(d);
        vec2 dd = vec2(cos(base_ang), sin(base_ang));
        vec2 ddir = vec2(dd.x * cr - dd.y * sr, dd.x * sr + dd.y * cr);
        float horizon = -1e9;
        for (int kk = 1; kk <= 32; kk++) {
            if (kk > K) break;
            float d_k = float(kk) * step_px;
            vec2 sp = p * (res * 0.5) + ddir * d_k;   // sample in pixel space
            vec2 suv = sp / res + 0.5;
            suv = clamp(suv, 0.0, 1.0);
            vec2 sc = (suv - 0.5) * 2.0 * max(u_freq, 0.5);
            float hq = _hbao_fbm(sc + off, int(clamp(u_octaves, 1.0, 6.0))) * 0.5 + 0.5;
            float phi = atan((hq - h) * k, d_k);
            horizon = max(horizon, phi);
        }
        horizon = clamp(horizon, 0.0, radians(85.0));
        vis_sum += 0.5 * (1.0 + cos(horizon));
    }
    float ao = vis_sum / float(D);

    // Radius-driven AO intensity gain (the honest, visible radius lever).
    float gain = clamp(0.4 + 0.05 * (u_radius - 4.0), 0.3, 4.0);
    ao = clamp(0.5 + (ao - 0.5) * gain, 0.0, 1.0);

    // ── Hillshade (Lambert from a movable light) ──
    float az = radians(u_light_az);
    float el = radians(u_light_el);
    if (amode == 3) az += t;   // rotate_light sweep
    vec3 L = vec3(cos(el) * cos(az), cos(el) * sin(az), sin(el));
    // surface normal from a cheap height gradient (finite differences)
    float e = 1.0 / max(res.x, res.y);
    vec2 scp = p * max(u_freq, 0.5);
    float hx = _hbao_fbm(scp + vec2(e, 0.0) * 50.0 + off, int(clamp(u_octaves, 1.0, 6.0))) * 0.5 + 0.5;
    float hy = _hbao_fbm(scp + vec2(0.0, e) * 50.0 + off, int(clamp(u_octaves, 1.0, 6.0))) * 0.5 + 0.5;
    vec3 nrm = normalize(vec3(-k * (hx - h) * 30.0, -k * (hy - h) * 30.0, 1.0));
    float shade = max(dot(nrm, L), 0.0);
    shade = u_ambient + (1.0 - u_ambient) * shade;

    // ── Compose ──
    int cmode = int(clamp(floor(u_colormode + 0.5), 0.0, 4.0));
    int mode = int(clamp(floor(u_mode + 0.5), 0.0, 2.0));  // 0 ao,1 shaded,2 height
    float disp;
    if (mode == 2) disp = h;
    else if (mode == 0) disp = clamp((ao - 0.5) * u_contrast + 0.5, 0.0, 1.0);
    else { float comb = clamp(ao * shade, 0.0, 1.0);
           disp = clamp((comb - 0.5) * u_contrast + 0.5, 0.0, 1.0); }

    vec3 rgb;
    if (cmode == 0) rgb = vec3(disp * 0.55, disp * 0.78, disp);          // steel
    else if (cmode == 1) rgb = vec3(disp, disp * 0.72, disp * 0.28);     // amber
    else if (cmode == 2) rgb = _hbao_inferno(disp);                     // inferno
    else if (cmode == 3) {                                               // spectral
        vec3 hsv = vec3(disp, clamp(0.25 + disp * 0.6, 0.0, 1.0), clamp(0.2 + disp * 0.9, 0.0, 1.0));
        rgb = _hbao_hsv2rgb(hsv);
    } else rgb = vec3(disp);                                             // grayscale
    f_color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
''',
          uniforms={
              "freq":        {"glsl": "float", "min": 1.0, "max": 12.0, "default": 5.0,
                             "description": "noise frequency of the height field"},
              "octaves":     {"glsl": "int", "min": 1, "max": 6, "default": 4,
                             "description": "fbm octaves"},
              "height_scale": {"glsl": "float", "min": 0.2, "max": 6.0, "default": 2.0,
                               "description": "world height per unit luminance (AO strength)"},
              "radius":      {"glsl": "float", "min": 4.0, "max": 64.0, "default": 24.0,
                              "description": "AO sampling radius in pixels"},
              "directions":  {"glsl": "int", "min": 3, "max": 16, "default": 8,
                              "description": "number of azimuth rays"},
              "steps":       {"glsl": "int", "min": 4, "max": 32, "default": 16,
                             "description": "horizon samples per ray"},
              "jitter":      {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
                             "description": "per-pixel disk rotation (breaks banding)"},
              "mode":        {"glsl": "choice", "choices": ["ao", "shaded", "height"],
                             "default": "shaded", "description": "output composition"},
              "light_az":    {"glsl": "float", "min": 0.0, "max": 360.0, "default": 135.0,
                             "description": "light azimuth in degrees"},
              "light_el":    {"glsl": "float", "min": 5.0, "max": 85.0, "default": 45.0,
                             "description": "light elevation in degrees"},
              "ambient":     {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.3,
                             "description": "ambient term of the hillshade combine"},
              "contrast":    {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.0,
                             "description": "tone contrast of the AO display"},
              "colormode":   {"glsl": "choice", "choices": ["steel", "amber", "inferno", "spectral", "grayscale"],
                             "default": "steel", "description": "output color"},
              "anim_mode":   {"glsl": "choice", "choices": ["none", "evolve", "drift", "rotate_light"],
                             "default": "none", "description": "animation mode"},
          }
          )