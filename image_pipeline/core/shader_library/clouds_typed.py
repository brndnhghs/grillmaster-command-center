"""clouds_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════
#  Volumetric Clouds (node 308) — ray-marched fbm density field
# ═══════════════════════════════════════════════════════════════
# Screen-space volumetric cloud render: march a fixed ray range through a
# world-Y slab, sample an fbm density field (advected by a wind vector driven
# by u_time), and accumulate single-scatter sunlight (a short light march
# toward the sun for self-shadowing) with Beer-Lambert absorption. Background
# is a sky gradient + sun glow. Closed-form f(uv,t) — no depth buffer needed,
# so it is a pure procedural twin (no input image).
_register("clouds_typed",
          "Raymarched volumetric clouds — screen-space fbm density march with "
          "single-scatter sun lighting over a sky gradient (typed, node 308)",
          "procedural", '''void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y;

    vec3 ro = vec3(0.0, 0.5, 3.0);
    vec3 rd = normalize(vec3(uv * 1.2, -1.6));

    float t = u_time * u_speed;
    vec3 wind = vec3(t * u_wind, 0.0, t * u_wind * 0.4);

    float az = u_sun_azim * 6.2831853;
    float el = mix(0.05, 1.2, u_sun_elev);
    vec3 sun = normalize(vec3(cos(az) * cos(el), sin(el), sin(az) * cos(el)));

    float h = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(u_sky_bottom, u_sky_top, h);
    sky += vec3(1.0, 0.92, 0.72) * pow(max(dot(rd, sun), 0.0), 12.0) * 0.5;

    float y0 = -1.0, y1 = 2.0;
    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    const int STEPS = 40;
    float tB = 6.0;
    float stepSize = tB / float(STEPS);

    for (int i = 0; i < STEPS; i++) {
        float tt = (float(i) + 0.5) * stepSize;
        vec3 pos = ro + rd * tt + wind;
        float vert = smoothstep(y0, y0 + 0.6, pos.y) * (1.0 - smoothstep(y1 - 0.6, y1, pos.y));
        float d = fbm(pos.xy * 0.6 + pos.z * 0.4 + 10.0);
        d = smoothstep(1.0 - u_coverage, 1.0, d) * vert;
        d *= u_density;
        if (d > 0.001) {
            float ls = 0.0;
            vec3 lp = pos;
            for (int j = 0; j < 4; j++) {
                lp += sun * 0.25;
                float ld = fbm((lp + wind).xy * 0.6 + (lp.z) * 0.4 + 10.0);
                ls += smoothstep(1.0 - u_coverage, 1.0, ld);
            }
            float lightT = exp(-ls * 0.5 * u_density);
            float absorb = exp(-d * stepSize * 3.0);
            vec3 sunCol = vec3(1.0, 0.95, 0.85);
            scattered += transmittance * (1.0 - absorb) * lightT * sunCol;
            transmittance *= absorb;
        }
        if (transmittance < 0.02) break;
    }

    vec3 col = sky * transmittance + scattered;
    col = pow(clamp(col, 0.0, 1.0), vec3(0.9));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "coverage":  {"glsl": "float", "min": 0.05, "max": 0.98, "default": 0.5,
                  "description": "cloud coverage threshold"},
    "density":   {"glsl": "float", "min": 0.0, "max": 2.5, "default": 1.0,
                  "description": "cloud density multiplier"},
    "wind":      {"glsl": "float", "min": 0.0, "max": 2.0, "default": 0.6,
                  "description": "wind speed (advection)"},
    "sun_elev":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                  "description": "sun elevation"},
    "sun_azim":  {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.25,
                  "description": "sun azimuth"},
    "speed":     {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.6,
                  "description": "animation speed"},
    "sky_top":   {"glsl": "color", "default": "#1a4a8a", "description": "zenith sky color"},
    "sky_bottom":{"glsl": "color", "default": "#cfe3f2", "description": "horizon sky color"},
})