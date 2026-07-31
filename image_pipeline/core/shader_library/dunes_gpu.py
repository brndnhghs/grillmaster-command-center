"""dunes_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dunes_gpu",
          "Sand dune migration height field (client-GPU twin of node 172)",
          "procedural", '''
void main() {
    // u_params.x = wind_strength (0.5 -> ~0.6 default),
    // u_params.y = sediment_supply (0.5 -> ~0.8 default),
    // u_params.z/.w unused (wind rotation + render style are procedural-only
    // live-preview defaults; CPU export is authoritative — two-tier precision).
    // Closed-form wave superposition of (uv, t) -> exact parity preview, no
    // seeded-layout divergence (same family as 125 Chladni / 164 Moiré).
    float wind = 0.1 + u_wind_strength * 1.0;     // ~0.6 at neutral 0.5
    float sed  = 0.1 + u_sediment_supply * 1.4;     // ~0.8 at neutral 0.5
    float t = u_time * 0.05;                 // matches node: t = frame*0.04

    // Slowly rotating wind (node "evolve" mode) -> migrating, merging field.
    float windAngle = t * 0.15;
    vec2 res = u_resolution;
    vec2 p = (v_uv - 0.5) * res;             // pixel-centered coords

    float h = 0.0;

    // ── Small ripples (10 waves spread around wind direction) ──
    for (int i = 0; i < 10; i++) {
        float fi = float(i);
        float amp = (0.22 / (fi + 1.0)) * wind;
        float wl = 8.0 + fi * 5.0;
        float aoff = (fi - 5.0) * 0.055;
        float ang = windAngle + aoff;
        vec2 d = vec2(cos(ang), sin(ang));
        float ph = fi * 1.3 + t * (0.006 + fi * 0.0006);
        float proj = (p.x * d.x + p.y * d.y) / wl + ph;
        h += amp * (sin(proj * 6.2831853) * 0.5 + 0.5);
    }

    // ── Large dune features (4 waves, longer wavelength) ──
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float amp = (0.30 + fi * 0.06) * sed;
        float wl = 55.0 + fi * 30.0;
        float wob = sin(t * 0.008 + fi * 2.0) * 0.15;
        float ang = windAngle + wob;
        vec2 d = vec2(cos(ang), sin(ang));
        float ph = fi * 2.5 + t * (0.003 + fi * 0.0004);
        float proj = (p.x * d.x + p.y * d.y) / wl + ph;
        h += amp * (sin(proj * 6.2831853) * 0.5 + 0.5);
    }

    // ── Subtle stochastic texture ──
    h += noise(p * 0.05 + vec2(t * 0.1)) * 0.15;

    // ── Normalize + hypsometric tint (matches node render_style=height) ──
    float hn = clamp(h * 0.55, 0.0, 1.0);
    vec3 col;
    if (hn < 0.20) {
        col = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), hn / 0.20);
    } else if (hn < 0.40) {
        col = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (hn - 0.20) / 0.20);
    } else if (hn < 0.60) {
        col = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (hn - 0.40) / 0.20);
    } else if (hn < 0.80) {
        col = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.25), (hn - 0.60) / 0.20);
    } else {
        col = mix(vec3(1.0, 0.5, 0.25), vec3(1.0, 0.96, 1.0), (hn - 0.80) / 0.20);
    }
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
    "wind_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "wind strength"},
    "sediment_supply": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "sediment supply"}
}
    )