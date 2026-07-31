"""acpm_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# Allen-Cahn + Perona-Malik anisotropic diffusion (node 146 "AC + PM Diffusion" twin).
# Single scalar field c in [-1,1], packed in .r. p1=alpha (diffusion strength),
# p2=K (PM edge sensitivity), p3=bias (constant double-well shift), p4=dt.
# Live-preview approximation of the CPU sim: omits per-frame noise + the time
# ramp on bias (CPU authoritative for export). Resuses the 5-pt ping-pong template.
_register("acpm_seed",
          "AC+PM seed: signed +/-1 blobs in .r (node 146 twin)",
          "procedural", '''
void main() {
    float c = 0.0;
    for (int i = 0; i < 24; i++) {
        float fi = float(i);
        vec2 ctr = vec2(hash21(vec2(fi + 0.5, 1.37)), hash21(vec2(fi + 0.5, 7.91)));
        ctr = 0.05 + 0.90 * ctr;
        float r = 0.03 + 0.05 * hash21(vec2(fi + 2.3, 4.1));
        float d = distance(v_uv, ctr);
        float signv = (mod(fi, 2.0) < 0.5) ? 1.0 : -1.0;
        c += signv * exp(-(d * d) / (r * r));
    }
    c = clamp(c, -1.0, 1.0);
    f_color = vec4(c, 0.0, 0.0, 1.0);
}
''')