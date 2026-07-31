"""colony_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# Bacterial colony (nodes 143, 160): nutrient N (.r), colony C (.g).
# growth of C where N present; consumption of N by C; diffusion of N.
# p1=growth, p2=diff_c, p3=consumption, p4=death.
_register("colony_seed",
          "Bacterial colony seed: nutrient full, colony disc at center (nodes 143/160 twin)",
          "procedural", '''
void main() {
    float N = 1.0;
    float d = distance(v_uv, vec2(0.5));
    float C = d < 0.06 ? 1.0 : 0.0;
    f_color = vec4(N, C, 0.0, 1.0);
}
''')