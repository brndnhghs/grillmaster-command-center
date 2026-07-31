"""phasor_noise_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO



# ── Node 361: Phasor Noise (Tricard et al. 2019) typed GPU twin of CPU node
# 1006. Sums complex Gaussian-windowed phasors over a 3x3 jittered-grid
# neighbourhood, extracts the ARGUMENT (constant-amplitude phase field), then
# feeds it through a periodic profile — reproducing the crisp fingerprint/
# wood-grain stripes of phasor noise. Closed-form f(uv,t): the global phase
# advances with u_time so the live preview genuinely animates (survives the
# contrast-only static-liveness cull). CPU numpy node 1006 stays
# authoritative for export (two-tier precision).
_register("phasor_noise_gpu",
          "Phasor Noise -- sparse-convolution complex-phasor field with a "
          "periodic profile (typed GPU twin of node 1006); anisotropic "
          "fingerprint/wood-grain stripes; genuinely animated (global phase "
          "advances with u_time)",
          "procedural", _INFERNO + '''
// per-cell random impulse: orientation angle, phase offset, frequency jitter
vec3 cell_impulse(vec2 cell) {
    float a = hash21(cell + 0.13) * 6.28318530718;          // orientation
    float psi = hash21(cell + 3.71) * 6.28318530718;        // phase offset
    float fj = 0.75 + 0.5 * hash21(cell + 7.42);            // freq jitter
    return vec3(a, psi, fj);
}

void main() {
    // Map uv into grid space; u_scale = cells across the canvas.
    float cells = max(2.0, u_scale);
    vec2 gp = v_uv * cells;
    vec2 base = floor(gp);

    float t = u_time * 0.6;

    // Accumulate the complex phasor sum over a 3x3 neighbourhood.
    vec2 acc = vec2(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 cell = base + vec2(float(dx), float(dy));
            // jittered impulse centre inside the cell
            vec2 jit = vec2(hash21(cell + 1.7), hash21(cell + 9.3));
            vec2 ctr = cell + jit;
            vec3 imp = cell_impulse(cell);
            float ang = imp.x;
            // optional domain warp bends orientation toward a smooth flow
            ang += u_anisotropy * (noise(cell * 0.5 + t * 0.1) - 0.5) * 3.14159265;
            float psi = imp.y;
            float fj = imp.z;

            vec2 d = gp - ctr;
            // oriented coordinate
            float ca = cos(ang), sa = sin(ang);
            float u = d.x * ca + d.y * sa;
            float vv = -d.x * sa + d.y * ca;
            // anisotropic Gaussian envelope (bandwidth <- u_falloff)
            float bw = max(0.2, u_falloff);
            float aM = bw;
            float am = bw / (1.0 + u_anisotropy * 4.0);
            float env = exp(-3.14159265 * (aM*aM*u*u + am*am*vv*vv));
            // windowed unit oscillation (frequency <- u_frequency)
            float theta = 6.28318530718 * u_frequency * fj * u + psi;
            acc += env * vec2(cos(theta), sin(theta));
        }
    }

    // phasor field = ARGUMENT of the complex sum (constant amplitude)
    float phase = atan(acc.y, acc.x);
    float ph = phase + (u_animate > 0.5 ? t : 0.0);

    // periodic profile
    int prof = int(clamp(u_profile + 0.5, 0.0, 3.0));
    float val;
    if (prof == 1) {              // sawtooth
        val = fract(ph / 6.28318530718 + 0.5);
    } else if (prof == 2) {       // square (smooth)
        float k = max(1.0, u_sharpness * 12.0);
        val = 0.5 + 0.5 * tanh(k * sin(ph));
    } else if (prof == 3) {       // triangle
        float saw = fract(ph / 6.28318530718 + 0.5);
        val = 1.0 - abs(2.0 * saw - 1.0);
    } else {                      // sine
        val = 0.5 + 0.5 * sin(ph);
    }

    // coherence fade where the phasor magnitude is tiny (phase undefined)
    float mag = length(acc);
    float coh = clamp(mag / 0.6, 0.0, 1.0);
    val = 0.5 + (val - 0.5) * coh;
    val = clamp(0.5 + (val - 0.5) * u_contrast, 0.0, 1.0);

    vec3 col = inferno(val);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "scale":      {"glsl": "float", "min": 2.0, "max": 40.0, "default": 12.0, "description": "grain feature count (cells across the canvas)"},
    "anisotropy": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.8, "description": "0 isotropic, 1 strongly directional streaks"},
    "falloff":    {"glsl": "float", "min": 0.3, "max": 1.6, "default": 0.7, "description": "Gaussian envelope bandwidth (higher = more compact kernels)"},
    "frequency":  {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.2, "description": "base stripe frequency (cycles per feature)"},
    "profile":    {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0, "description": "periodic profile: 0 sine, 1 sawtooth, 2 square, 3 triangle"},
    "sharpness":  {"glsl": "float", "min": 0.1, "max": 2.0, "default": 0.6, "description": "edge sharpness for the square profile"},
    "contrast":   {"glsl": "float", "min": 0.5, "max": 3.0, "default": 1.2, "description": "final tone contrast"},
    "animate":    {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0, "description": "advance global phase with time (0 static, 1 glide)"},
})