"""plasma_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("plasma_gpu",
          "Plasma fractal heightfield (client-GPU twin of node 31)",
          "procedural", '''
void main() {
    // Typed uniforms match node 31's real numeric params (pitfall #14):
    //   u_size          plasma grid size (64..1024) -> spatial frequency
    //   u_roughness     initial roughness amplitude (0.05..2) -> warp strength
    //   u_octaves       fBm octaves (1..6) -> detail iterations
    //   u_seed_strength (0..1) -> heightfield seed blend
    // Closed-form diamond-square approximation: domain-warped fBm heightfield
    // with a cosine plasma palette. CPU numpy node stays authoritative for
    // export. Uses prologue helpers noise/fbm (no local redefinition).
    vec2 uv = v_uv;
    float t = u_time * 0.15;                 // height_warp-style evolution

    float freq = u_size * 0.012;             // 512 -> ~6.1 cycles across
    vec2 q = uv * freq;

    // animated domain warp (roughness drives amplitude)
    float warp = u_roughness * 0.6;
    q += warp * vec2(fbm(q + vec2(0.0, t)),
                     fbm(q + vec2(5.2, 1.3 - t)));

    // fBm accumulation (constant loop bound, break on octave count)
    float h = 0.0;
    float amp = 0.5;
    float fr = 1.0;
    int oct = int(clamp(u_octaves, 1.0, 6.0));
    for (int i = 0; i < 6; i++) {
        if (i >= oct) break;
        h += amp * fbm(q * fr + vec2(float(i) * 1.7, t * 0.5));
        fr *= 2.0;
        amp *= 0.5;
    }
    h = mix(h, fract(h + u_seed_strength), 0.5);
    h = clamp(h * 0.5 + 0.5, 0.0, 1.0);

    // cosine plasma palette + height shading
    vec3 col = 0.5 + 0.5 * cos(6.2831853 * (h + vec3(0.0, 0.33, 0.67)));
    col *= 0.6 + 0.4 * h;
    f_color = vec4(col, 1.0);
}
''',
    uniforms={
        "size": {"glsl": "float", "min": 64.0, "max": 1024.0, "default": 512.0,
                 "description": "plasma grid size (frequency)"},
        "roughness": {"glsl": "float", "min": 0.05, "max": 2.0, "default": 0.5,
                      "description": "domain-warp strength"},
        "octaves": {"glsl": "float", "min": 1.0, "max": 6.0, "default": 3.0,
                    "description": "fBm octaves"},
        "seed_strength": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.6,
                          "description": "heightfield seed blend"}
    }
    )