"""grav_lens_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ══════════════════════════════════════════════════════════════════════════
#  CPU-node closed-form procedural twins (Route 0 / GPU-First gap mirror)
#  Live-preview GPU twins for three pattern/math_art nodes whose CPU algorithm
#  is a faithful per-pixel closed-form f(uv, t) generator with NO close
#  existing twin. CPU node stays authoritative for export; these drive the
#  client-side live preview. Every numeric CPU param becomes a named u_<name>
#  uniform/SCALAR port (typed-uniform contract). Choice params (palette / mode
#  / pattern / color_mode) are dropped to GPU_PREVIEW_DROP_ALLOW — the twins
#  animate continuously from u_time so the preview is always live, and the CPU
#  export honours the exact choices. Palettes are inlined (no late-helper
#  _INFERNO ordering pitfall). Animation uses cos()/linear terms so the
#  t=0 vs t=pi audit is never a sin-phase false negative.
# ══════════════════════════════════════════════════════════════════════════

# ── 995 Gravitational Lensing / Einstein Ring (client-GPU twin) ──
_register("grav_lens_gpu",
          "Gravitational Lensing Einstein Ring (client-GPU twin of node 995)",
          "procedural",
'''void main() {
    // Thin-lens deflection: beta = theta * (1 - thetaE^2/|theta|^2). Sample a
    // procedural sky (fbm nebula + hashed stars) at the mapped source position
    // and brighten by the magnification mu. Continuous drift keeps it live.
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    uv *= 2.2;                                    // world scale ~ CPU node
    float t = u_time * u_anim_speed;

    float tE = u_einstein_radius;
    // drift the source in a slow circle (matches CPU 'drift' mode)
    vec2 p = uv + 0.25 * vec2(sin(t), cos(t));
    // breathe the lens mass a touch (cos, no 0/pi degeneracy)
    tE *= 0.85 + 0.15 * cos(t * 0.7);

    float r2 = dot(p, p) + 1e-4;
    float inv = (tE * tE) / r2;
    vec2 beta = p * (1.0 - inv);                  // source position
    float mu = 1.0 / abs(1.0 - inv * inv);
    mu = clamp(mu, 1.0, 8.0);

    // procedural sky at beta
    vec2 sky_uv = beta * (2.0 + u_neb_scale * 0.6);
    float neb = pow(clamp(fbm(sky_uv + vec2(0.0, t * 0.05)) , 0.0, 1.0), 1.6) * u_nebula;
    float star = smoothstep(1.0 - u_star_density * 40.0, 1.0, hash21(floor(beta * 240.0)));
    star += 0.5 * smoothstep(0.985, 1.0, hash21(floor(beta * 90.0 + 7.0)));

    // palette (cosmic default, inlined)
    vec3 neb_rgb = vec3(0.32, 0.22, 0.55);
    vec3 star_rgb = vec3(0.80, 0.86, 1.0);
    vec3 sky = neb_rgb * neb + star_rgb * star;

    float rr = sqrt(r2);
    float ring = exp(-((rr - tE) * (rr - tE)) / (2.0 * u_ring_width * u_ring_width));
    vec3 glow = star_rgb * ring * u_ring_brightness;

    vec3 col = clamp(sky * mu + glow, 0.0, 1.0);
    col = clamp(col * u_exposure, 0.0, 1.0);
    f_color = vec4(col, 1.0);
}
''',
uniforms={
    "einstein_radius": {"glsl": "float", "min": 0.05, "max": 0.9, "default": 0.35, "description": "Einstein radius (lens mass, ring size)"},
    "star_density":    {"glsl": "float", "min": 0.0005, "max": 0.02, "default": 0.004, "description": "fraction of pixels that are stars"},
    "nebula":          {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "nebula cloud intensity"},
    "neb_scale":       {"glsl": "float", "min": 1.0, "max": 8.0, "default": 3.0, "description": "nebula fbm frequency"},
    "exposure":        {"glsl": "float", "min": 0.2, "max": 4.0, "default": 1.4, "description": "output brightness multiplier"},
    "ring_brightness": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.2, "description": "Einstein-ring glow strength"},
    "ring_width":      {"glsl": "float", "min": 0.01, "max": 0.3, "default": 0.06, "description": "Einstein-ring glow width"},
    "anim_speed":      {"glsl": "float", "min": 0.1, "max": 5.0, "default": 1.0, "description": "animation speed"},
})