"""lv3_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# 3-species Lotka-Volterra (node 120): U,V,W in r,g,b — cyclic predation.
# p1,p2,p3,p4 = interaction strengths (live preview approx; CPU authoritative).
_register("lv3_seed",
          "3-species LV seed: U~1, V~0.5, W~0.5 with hashed blobs (node 120 twin)",
          "procedural", '''
void main() {
    float U = 1.0, V = 0.5, W = 0.5;
    for (int i = 0; i < 14; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi+0.5,1.37)), hash21(vec2(fi+0.5,7.91)));
        c = 0.05 + 0.90*c; float d = distance(v_uv, c);
        float blob = exp(-(d*d)/0.002);
        U -= 0.4*blob; V += 0.5*blob; W += 0.3*blob;
    }
    f_color = vec4(clamp(U,0.0,1.0), clamp(V,0.0,1.0), clamp(W,0.0,1.0), 1.0);
}
''')