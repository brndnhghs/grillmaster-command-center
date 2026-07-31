"""thin_film_spectral_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _TF_HELPERS



_register("thin_film_spectral_gpu",
          "Spectral thin-film interference iridescence (client-GPU twin of node 1004)",
          "procedural",
          _TF_HELPERS + '''
void main() {
    // ── Named uniforms match node 1004's REAL params ──
    // thickness, thickness_range, ior, drainage, view_angle, brightness,
    // anim_speed + the choice param anim_mode (none/flow/swirl/pulse).
    vec2 res = u_resolution;
    vec2 uv = v_uv;
    float mx = max(res.x, res.y);
    // Match the CPU coordinate frame: u = x/max, v = y/max in [0, ~1].
    float u = uv.x * res.x / mx;
    float v = (1.0 - uv.y) * res.y / mx;   // flip: CPU row 0 is the top

    int amode = int(clamp(floor(u_anim_mode + 0.5), 0.0, 3.0)); // 0 none,1 flow,2 swirl,3 pulse
    float t = (amode == 0) ? 0.0 : u_time * max(u_anim_speed, 0.0);

    // ── Thickness field (procedural fbm bands + drainage) ──
    float cx = 0.5, cy = 0.5;
    float dx = u - cx, dy = v - cy;
    if (amode == 2) {           // swirl: rotate the sample frame
        float ca = cos(t), sa = sin(t);
        float rx = ca * dx - sa * dy;
        float ry = sa * dx + ca * dy;
        dx = rx; dy = ry;
    }
    float fx = dx + (amode == 1 ? t : 0.0);   // flow: bands travel in x
    float fy = dy;
    float scale = 4.0;          // CPU uses 3..5 (rng); twin fixes a mid value
    float h = _tf_fbm(vec2(fx * scale * 6.0, fy * scale * 6.0));
    h = 0.5 + 0.5 * h;
    // Drainage: film thins toward the top (v small at top).
    h = clamp(h - u_drainage * (0.5 - v), 0.0, 1.0);
    float thickness01 = clamp(h, 0.0, 1.0);

    // pulse: thickness range breathes (smooth offset sine, no cusp).
    float trange = u_thickness_range;
    if (amode == 3) trange *= (0.1 + 0.9 * (0.5 + 0.5 * sin(t)));

    // ── View-angle cos(theta_t) via Snell (dome normal tilt) ──
    float nx = (uv.x - 0.5) * u_view_angle;
    float ny = (uv.y - 0.5) * u_view_angle;
    float sin_i = clamp(sqrt(nx * nx + ny * ny), 0.0, 0.999);
    float cosT = clamp(sqrt(clamp(1.0 - (sin_i * sin_i) / (u_ior * u_ior), 1e-4, 1.0)), 0.05, 1.0);

    // ── Spectral interference integral against CIE CMF ──
    float d_nm = u_thickness + trange * (thickness01 - 0.5);
    vec3 xyz = vec3(0.0);
    vec3 white = vec3(0.0);
    const float PI = 3.14159265;
    for (int k = 0; k < 35; k++) {
        float lam = 380.0 + float(k) * 10.0;   // 380..720 nm, 10 nm step
        vec3 cmf = _tf_cmf(lam);
        float delta = (4.0 * PI * u_ior * d_nm * cosT) / lam + PI;
        float Rk = 0.5 * (1.0 + cos(delta));
        xyz += cmf * Rk;
        white += cmf;
    }
    xyz /= max(white.y, 1e-6);
    xyz *= u_brightness;
    vec3 rgb = _tf_xyz2srgb(xyz);
    f_color = vec4(rgb, 1.0);
}
''',
          uniforms={
              "thickness":       {"glsl": "float", "min": 100.0, "max": 1200.0, "default": 380.0,
                                  "description": "base film thickness (nm); dominant colour = d*n"},
              "thickness_range": {"glsl": "float", "min": 0.0, "max": 900.0, "default": 420.0,
                                  "description": "thickness variation (nm) — drives the colour bands"},
              "ior":             {"glsl": "float", "min": 1.05, "max": 2.5, "default": 1.33,
                                  "description": "film refractive index (soap 1.33, oil 1.45)"},
              "drainage":        {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.35,
                                  "description": "vertical thinning gradient (drains toward top)"},
              "view_angle":      {"glsl": "float", "min": 0.0, "max": 1.2, "default": 0.5,
                                  "description": "surface tilt (rad) — oblique edge colour shift"},
              "brightness":      {"glsl": "float", "min": 0.2, "max": 1.5, "default": 0.9,
                                  "description": "overall reflected intensity scale"},
              "anim_speed":      {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                                  "description": "animation speed multiplier"},
              "anim_mode":       {"glsl": "choice", "choices": ["none", "flow", "swirl", "pulse"],
                                  "default": "none", "description": "animation mode"},
          }
          )