"""hex_tiling_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# ── Stochastic hex-tiling (Heitz & Neyret, HPG 2018) ──
# "High-Performance By-Example Noise using a Histogram-Preserving Blending
# Operator" — tiles a texture across the plane with NO visible repetition.
# Ref: Heitz & Neyret, ACM SIGGRAPH/HPG 2018, doi:10.1145/3233310
# Core idea: overlay a triangle grid (hex lattice); at each pixel the three
# nearest lattice vertices each provide a randomly-offset sample of the input;
# blend them with barycentric weights raised to a contrast exponent so the
# variance-preserving mix hides the tiling seams. The random per-vertex
# translation breaks the lattice periodicity → aperiodic tiling of one image.
_register("hex_tiling_gpu",
          "Stochastic hex-tiling — repetition-free texture tiling (Heitz-Neyret 2018)",
          "filter", _filter_typed('''
    float scale = u_scale;            // tiles across the image
    float contrast = u_contrast;      // blend sharpness (variance preserve)
    float randomness = u_randomness;  // per-tile random offset strength

    vec2 st = uv * scale;
    // skew into triangle-lattice space (hex lattice basis)
    mat2 UNSKEW = mat2(1.0, 0.0, -0.5773503, 1.1547005);
    vec2 sp = UNSKEW * st;
    vec2 baseId = floor(sp);
    vec2 f = fract(sp);
    // pick one of the two triangles of the rhombus by the diagonal
    vec2 id1, id2, id3;
    vec3 w;
    if (f.x + f.y < 1.0) {
        id1 = baseId;
        id2 = baseId + vec2(1.0, 0.0);
        id3 = baseId + vec2(0.0, 1.0);
        w = vec3(1.0 - f.x - f.y, f.x, f.y);
    } else {
        id1 = baseId + vec2(1.0, 1.0);
        id2 = baseId + vec2(1.0, 0.0);
        id3 = baseId + vec2(0.0, 1.0);
        w = vec3(f.x + f.y - 1.0, 1.0 - f.y, 1.0 - f.x);
    }
    // hash each vertex id -> random per-vertex uv translation (breaks periodicity)
    vec2 h1 = fract(sin(vec2(dot(id1, vec2(127.1, 311.7)), dot(id1, vec2(269.5, 183.3)))) * 43758.5453);
    vec2 h2 = fract(sin(vec2(dot(id2, vec2(127.1, 311.7)), dot(id2, vec2(269.5, 183.3)))) * 43758.5453);
    vec2 h3 = fract(sin(vec2(dot(id3, vec2(127.1, 311.7)), dot(id3, vec2(269.5, 183.3)))) * 43758.5453);
    vec4 c1 = texture(u_texture, uv + (h1 - 0.5) * randomness);
    vec4 c2 = texture(u_texture, uv + (h2 - 0.5) * randomness);
    vec4 c3 = texture(u_texture, uv + (h3 - 0.5) * randomness);
    // contrast-preserving (variance) blend: sharpen the bary weights so the
    // dominant tile wins near vertices -> hides seams (Heitz-Neyret operator).
    vec3 wp = pow(max(w, vec3(0.0)), vec3(1.0 + contrast * 6.0));
    wp /= max(dot(wp, vec3(1.0)), 1e-4);
    vec4 blended = c1 * wp.x + c2 * wp.y + c3 * wp.z;
    f_color = vec4(blended.rgb, 1.0);
'''), uniforms={
    "scale":      {"glsl": "float", "min": 1.0, "max": 12.0, "default": 4.0, "description": "tiles across image"},
    "contrast":   {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "seam-hiding blend sharpness"},
    "randomness": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.4, "description": "per-tile random offset"},
})