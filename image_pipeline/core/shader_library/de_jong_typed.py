"""de_jong_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# ── Node 332: De Jong Attractor — GPU live-preview twin of CPU node 498 ──
# De Jong recurrence x'=sin(a·y)−cos(b·x), y'=sin(c·x)−cos(d·y). The CPU node
# iterates many parallel walkers, splats a density grid, and tone-maps with an
# inferno ramp; this twin does the SAME in a single fragment pass (a coarse
# accumulation grid + 3×3 gaussian splat) so the live preview is a faithful
# parity of the density (inferno) colouring. Named typed uniforms mirror node
# 498's real numeric params (a/b/c/d/exposure) — contract #5/#6. `morph`+`speed`
# animate the parameters via u_time (matches CPU anim_mode="morph_all"), so the
# live preview is genuinely time-varying and survives the contrast-only static
# liveness cull. The CPU numpy node stays authoritative for export; this is the
# live GPU source for the De Jong category (categorical coverage gap: 498 had
# no GPU twin until now).
_register("de_jong_typed", "De Jong attractor density field (typed, node 498)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.05 * u_speed;
    float a = u_a, b = u_b, c = u_c, d = u_d;
    if (u_morph > 0.5) {
        // morph_all: smoothly perturb all four params (no abs(sin) cusp)
        a += 0.9 * sin(t);
        b += 0.9 * cos(t * 0.9);
        c += 0.9 * sin(t * 1.1 + 1.0);
        d += 0.9 * cos(t * 0.8 + 2.0);
    }
    const int ACC = 40;      // accumulation starts (parallel trajectories)
    const int STP = 64;      // points splatted per start
    const int GRID = 40;     // density accumulation grid (coarse, cheap)
    float dens[GRID * GRID];
    for (int k = 0; k < GRID * GRID; k++) dens[k] = 0.0;
    float span = 2.0 + max(max(abs(a), abs(b)), max(abs(c), abs(d)));
    for (int i = 0; i < ACC; i++) {
        float seed = float(i) / float(ACC);
        vec2 p = vec2(sin(seed * 12.9) * 1.7, cos(seed * 7.3) * 1.7);
        for (int k = 0; k < 8; k++) {            // transient discard
            p = vec2(sin(a * p.y) - cos(b * p.x),
                     sin(c * p.x) - cos(d * p.y));
        }
        for (int s = 0; s < STP; s++) {
            p = vec2(sin(a * p.y) - cos(b * p.x),
                     sin(c * p.x) - cos(d * p.y));
            vec2 g = clamp(p / span * 0.5 + 0.5, 0.0, 0.999);
            int gx = int(g.x * float(GRID));
            int gy = int(g.y * float(GRID));
            dens[gy * GRID + gx] += 1.0;
        }
    }
    // gaussian-splat the accumulation grid at the pixel position
    vec2 g = clamp(uv / span * 0.5 + 0.5, 0.0, 0.999);
    float fx = g.x * float(GRID) - 0.5;
    float fy = g.y * float(GRID) - 0.5;
    float acc = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float cx = floor(fx) + float(dx);
            float cy = floor(fy) + float(dy);
            if (cx < 0.0 || cy < 0.0 || cx >= float(GRID) || cy >= float(GRID)) continue;
            int idx = int(cy) * GRID + int(cx);
            float dist = length(vec2(fx, fy) - vec2(cx + 0.5, cy + 0.5));
            acc += dens[idx] * exp(-dist * dist * u_sharp);
        }
    }
    float glow = 1.0 - exp(-u_exposure * acc * u_density_scale);
    vec3 col = inferno(clamp(glow, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "a":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": -2.0,
             "description": "de Jong parameter a (shape control)"},
    "b":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": -2.0,
             "description": "de Jong parameter b (shape control)"},
    "c":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": -1.2,
             "description": "de Jong parameter c (shape control)"},
    "d":    {"glsl": "float", "min": -3.0, "max": 3.0, "default": 2.0,
             "description": "de Jong parameter d (shape control)"},
    "morph": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0,
              "description": "morph a/b/c/d over time (0=static, 1=morph_all)"},
    "exposure": {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.6,
                 "description": "tone-map exposure (glow brightness)"},
    "sharp": {"glsl": "float", "min": 1.0, "max": 60.0, "default": 18.0,
              "description": "splat tightness (band tightness)"},
    "density_scale": {"glsl": "float", "min": 0.1, "max": 3.0, "default": 1.0,
                      "description": "density accumulation normaliser"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "animation speed"},
})