"""nishita_sky_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── GPU-First categorical coverage (2026-07-14): Nishita atmospheric
# single-scattering sky — the GPU live-preview twin of CPU node 471. Closed-form
# per-pixel ray-march (no ping-pong state) so it verifies headlessly. Works in
# km units so the scale-height exponentials stay in a fp32-safe range (the CPU
# node's metre units would underflow exp() in fp32). The sun rides a day-arc
# driven by u_time (which advances one step per frame; pace is set by the
# graph's wired drivers/substeps, not a time_scale multiplier), so the live
# preview animates; sun_elevation/sun_azimuth are the base pose the graph can
# also wire
# CHOP generators into. CPU numpy node stays authoritative for export. ──
_register("nishita_sky_gpu",
          "Nishita atmospheric single-scattering sky (GPU twin of CPU node 471) "
          "— Rayleigh + Mie scattering integrated along view and light rays through "
          "a spherical atmosphere; animated sun day-arc via u_time.",
          "procedural", '''void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y;
    float fov = radians(u_fov);
    float f = tan(fov * 0.5);
    vec3 rd = normalize(vec3(uv.x * f, uv.y * f, -1.0));
    // Camera 1 km above the ground, inside the spherical atmosphere (units: km).
    vec3 ro = vec3(0.0, 6361.0, 0.0);

    // Animated sun: a day-arc around the node's base elevation/azimuth.
    // u_time advances one step per frame (pace via wired drivers/substeps),
    // so the live preview moves smoothly.
    float baseEl = radians(u_sun_elevation);
    float baseAz = radians(u_sun_azimuth);
    float el = clamp(baseEl + sin(u_time) * radians(40.0), -1.4, 1.55);
    float az = baseAz + u_time * 0.25;
    vec3 sun = normalize(vec3(cos(el) * sin(az), sin(el), -cos(el) * cos(az)));

    float Rg = 6360.0, Rt = 6420.0, Hr = 8.0, Hm = 1.2;

    // Ray vs top-of-atmosphere and ground spheres (a = 1).
    float b = dot(ro, rd);
    float cT = dot(ro, ro) - Rt * Rt;
    float tTop = -b + sqrt(max(b * b - cT, 0.0));
    float cG = dot(ro, ro) - Rg * Rg;
    float discG = b * b - cG;
    float tGround = -b - sqrt(max(discG, 0.0));
    bool ground = (discG >= 0.0) && (tGround > 0.0);
    float tMax = ground ? min(tTop, tGround) : tTop;
    tMax = max(tMax, 0.0);

    // Phase functions (view/sun angle).
    float cosT = dot(rd, sun);
    float phaseR = 3.0 / (16.0 * 3.14159265) * (1.0 + cosT * cosT);
    float g = 0.76;
    float denom = (2.0 + g * g) * pow(1.0 + g * cosT, 2.0);
    float phaseM = 3.0 / (8.0 * 3.14159265) * (1.0 - g * g) * (1.0 + cosT * cosT) / max(denom, 1e-8);

    vec3 betaR = vec3(5.8e-3, 13.5e-3, 33.1e-3) * u_rayleigh_k;
    vec3 betaM = vec3(21e-3) * u_mie_k;
    float SUN_I = 20.0;

    const int NS = 16, NL = 8;
    float seg = tMax / float(NS);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    vec3 lastAtten = vec3(1.0);
    float odr = 0.0, odm = 0.0;

    for (int i = 0; i < NS; i++) {
        float t = (float(i) + 0.5) * seg;
        vec3 p = ro + t * rd;
        float alt = length(p) - Rg;
        float hr = exp(-alt / Hr) * seg;
        float hm = exp(-alt / Hm) * seg;
        odr += hr; odm += hm;
        // Light ray optical depth toward the sun.
        float bL = dot(p, sun);
        float cL = dot(p, p) - Rt * Rt;
        float tTopL = -bL + sqrt(max(bL * bL - cL, 0.0));
        float segL = max(tTopL, 0.0) / float(NL);
        float odlr = 0.0, odlm = 0.0;
        for (int j = 0; j < NL; j++) {
            float tl = (float(j) + 0.5) * segL;
            vec3 pl = p + tl * sun;
            float altL = length(pl) - Rg;
            odlr += exp(-altL / Hr) * segL;
            odlm += exp(-altL / Hm) * segL;
        }
        vec3 tau = betaR * (odr + odlr) + betaM * (1.1 * (odm + odlm));
        vec3 atten = exp(-tau);
        lastAtten = atten;
        sumR += atten * hr * betaR * phaseR;
        sumM += atten * hm * betaM * phaseM;
    }

    vec3 col = (sumR + sumM) * SUN_I;
    // Sun disk + soft halo, modulated by transmittance toward the sun.
    float ang = acos(clamp(cosT, -1.0, 1.0));
    float rDisk = radians(u_sun_disk_radius);
    float disk = clamp(1.0 - ang / rDisk, 0.0, 1.0);
    float halo = clamp(1.0 - ang / (rDisk * 4.0), 0.0, 1.0) * 0.3;
    float glow = disk + halo;
    float trans = (lastAtten.r + lastAtten.g + lastAtten.b) / 3.0;
    col += glow * trans * vec3(1.0, 0.95, 0.85) * 20.0;
    // Tonemap + clamp.
    col = vec3(1.0) - exp(-col * u_exposure);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
''', uniforms={
    "sun_elevation":   {"glsl": "float", "min": -10.0, "max": 90.0, "default": 6.0,
                        "description": "sun elevation in degrees (base for the animated day-arc)"},
    "sun_azimuth":     {"glsl": "float", "min": 0.0, "max": 360.0, "default": 90.0,
                        "description": "sun azimuth in degrees"},
    "rayleigh_k":      {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                        "description": "Rayleigh (molecular) scattering strength"},
    "mie_k":           {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                        "description": "Mie (aerosol) scattering strength"},
    "exposure":        {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                        "description": "tonemap exposure"},
    "fov":             {"glsl": "float", "min": 20.0, "max": 120.0, "default": 60.0,
                        "description": "vertical field of view in degrees"},
    "sun_disk_radius": {"glsl": "float", "min": 0.3, "max": 5.0, "default": 1.2,
                        "description": "apparent sun disk radius in degrees"},
})


# ── 471 Nishita Atmospheric Sky (client-GPU twin of node 471) ─────────────────
# Faithful GLSL port of the CPU node's single-scattering Nishita model
# (Nishita et al. 1993): for each pixel we march the view ray through a
# spherical shell atmosphere accumulating in-scattered sunlight from Rayleigh
# (molecular, blue) + Mie (aerosol, sun halo) scattering, with a second march
# toward the sun for optical depth (horizon reddening). This is O(W·H) with a
# small fixed sample budget = the same cheap closed-form f(uv,t) profile as the
# CPU node, so it is an honest live-preview twin (CPU numpy node 471 stays the
# authoritative export source). All numeric params are REAL node-471 params,
# wired by name via the typed param_map (contract #5). No string/choice params
# (anim_mode) are mapped — pitfall #14.
_register("nishita_sky_gpu", "Nishita atmospheric scattering sky (client-GPU twin of node 471)",
          "procedural", '''
const float R_GROUND = 6360e3;
const float R_TOP    = 6420e3;
const float H_R      = 7994.0;
const float H_M      = 1200.0;
const float PI       = 3.14159265;
const vec3  BETA_R   = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const vec3  BETA_M   = vec3(21.0e-6, 21.0e-6, 21.0e-6);
const float MIE_G    = 0.76;
const float SUN_I    = 20.0;

// ray vs sphere centered at origin; returns (t_near, t_far)
vec2 _rsph(vec3 ro, vec3 rd, float R) {
    float b = 2.0 * dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float disc = max(b * b - 4.0 * c, 0.0);
    float sq = sqrt(disc);
    return vec2((-b - sq) * 0.5, (-b + sq) * 0.5);
}

// in-scatter along one view ray for a normalized sun direction sd.
vec3 _sky(vec3 rd, vec3 sd, float rk, float mk, float exposure,
          float fov, float disk_deg) {
    vec3 ro = vec3(0.0, R_GROUND + 1000.0, 0.0);
    vec2 a = _rsph(ro, rd, R_TOP);
    vec2 g = _rsph(ro, rd, R_GROUND);
    float ground = step(0.0, g.x);
    float t_max = (ground > 0.5) ? min(a.y, g.x) : a.y;
    t_max = max(t_max, 0.0);

    float ct = dot(rd, sd);
    float phase_r = 3.0 / (16.0 * PI) * (1.0 + ct * ct);
    float denom = (2.0 + MIE_G * MIE_G) * (1.0 + MIE_G * ct) * (1.0 + MIE_G * ct);
    float phase_m = 3.0 / (8.0 * PI) * (1.0 - MIE_G * MIE_G) * (1.0 + ct * ct)
                    / max(denom, 1e-8);

    float ns = 16.0, nsl = 8.0;
    float seg = t_max / ns;
    vec3 sum_r = vec3(0.0), sum_m = vec3(0.0);
    vec3 last_atten = vec3(1.0);
    for (int i = 0; i < 16; i++) {
        float t = (float(i) + 0.5) * seg;
        vec3 p = ro + t * rd;
        float hgt = length(p) - R_GROUND;
        float hr = exp(-hgt / H_R) * seg;
        float hm = exp(-hgt / H_M) * seg;
        // light march toward sun
        float tl = max(_rsph(p, sd, R_TOP).y, 0.0) / nsl;
        float tl_i = 0.5 * tl;
        float odlr = 0.0, odlm = 0.0;
        for (int j = 0; j < 8; j++) {
            vec3 pl = p + tl_i * sd;
            float hl = length(pl) - R_GROUND;
            odlr += exp(-hl / H_R) * tl;
            odlm += exp(-hl / H_M) * tl;
            tl_i += tl;
        }
        vec3 tau = BETA_R * (hr + odlr * rk) + BETA_M * (hm + odlm * 1.1 * mk);
        vec3 atten = exp(-tau);
        last_atten = atten;
        sum_r += atten * hr * BETA_R * rk * phase_r;
        sum_m += atten * hm * BETA_M * mk * phase_m;
    }
    vec3 col = (sum_r + sum_m) * SUN_I;
    float ang = acos(clamp(ct, -1.0, 1.0));
    float r_disk = radians(disk_deg);
    float disk = clamp(1.0 - ang / r_disk, 0.0, 1.0);
    float halo = clamp(1.0 - ang / (r_disk * 4.0), 0.0, 1.0) * 0.3;
    float glow = disk + halo;
    float trans = (last_atten.r + last_atten.g + last_atten.b) / 3.0;
    vec3 sun_tint = vec3(1.0, 0.95, 0.85);
    col += glow * trans * sun_tint * 20.0;
    col = 1.0 - exp(-col * exposure);
    return clamp(col, 0.0, 1.0);
}

void main() {
    // pixel -> view dir (camera looks down -Z, y up), fov in degrees
    vec2 uv = v_uv * 2.0 - 1.0;
    float aspect = u_resolution.x / u_resolution.y;
    float f = tan(radians(u_fov) * 0.5);
    vec3 rd = normalize(vec3(uv.x * aspect * f, uv.y * f, -1.0));
    // sun dir from elevation/azimuth (degrees); time glides elevation
    float el = radians(u_sun_elevation + u_time * 30.0);
    float az = radians(u_sun_azimuth);
    vec3 sd = vec3(cos(el) * sin(az), sin(el), -cos(el) * cos(az));
    rd = normalize(rd);
    sd = normalize(sd);
    vec3 col = _sky(rd, sd, u_rayleigh_k, u_mie_k, u_exposure, u_fov, u_sun_disk_radius);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "sun_elevation":   {"glsl": "float", "min": -10.0, "max": 90.0, "default": 6.0,
                        "description": "sun elevation (deg); time glides it for preview"},
    "sun_azimuth":     {"glsl": "float", "min": 0.0, "max": 360.0, "default": 90.0,
                        "description": "sun azimuth (deg)"},
    "rayleigh_k":      {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0,
                        "description": "Rayleigh scattering strength"},
    "mie_k":           {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0,
                        "description": "Mie (aerosol/haze) strength"},
    "exposure":        {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                        "description": "tonemap exposure"},
    "fov":             {"glsl": "float", "min": 20.0, "max": 120.0, "default": 60.0,
                        "description": "vertical field-of-view (deg)"},
    "sun_disk_radius": {"glsl": "float", "min": 0.3, "max": 5.0, "default": 1.2,
                        "description": "apparent sun disk radius (deg)"},
})