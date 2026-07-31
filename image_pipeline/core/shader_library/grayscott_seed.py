"""grayscott_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




_register("grayscott_seed",
          "Gray-Scott initial state: U=1, V=hashed seed blobs (client-GPU sim of node 155)",
          "procedural", '''
void main() {
    // U (substrate) starts ~1 everywhere; V (activator) as a few blobs, echoing
    // the CPU node's n_seeds gaussian patches (positions from a stable hash — the
    // GPU twin cannot reproduce numpy's PCG64 layout; it just needs live seeds).
    float U = 1.0;
    float V = 0.0;
    for (int i = 0; i < 20; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi + 0.5, 1.37)),
                      hash21(vec2(fi + 0.5, 7.91)));
        c = 0.05 + 0.90 * c;                 // keep blobs off the very edge
        float d = distance(v_uv, c);
        V += 0.5 * exp(-(d * d) / 0.0016);
    }
    V = clamp(V, 0.0, 1.0);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')