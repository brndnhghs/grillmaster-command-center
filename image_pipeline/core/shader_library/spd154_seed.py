"""spd154_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 154: Continuous Spatial PD (replicator dynamics) — PDE field ──
_register("spd154_seed",
          "CSPD #154 seed: hashed continuous strategy field s∈[0,1] (R=raw, G=trail)",
          "procedural", '''
void main() {
    float x = hash21(v_uv * u_resolution + 0.321);
    f_color = vec4(x * 0.6, x * 0.6, 0.0, 1.0);  // R=s, G=accum trail
}
''')