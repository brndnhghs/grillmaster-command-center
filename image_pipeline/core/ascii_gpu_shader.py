"""
Build the ascii_art_gpu GLSL shader source from precomputed shape data.

Uses the shape-vector approach from "ASCII characters are not pixels" (Alex Harri):
  - 6D staggered sampling circles per cell for shape-aware character selection
  - Directional + global contrast enhancement for sharp edges
  - Nearest-neighbor lookup in 6D shape space
  - Precomputed 4x5 glyph bitmaps for rendering
"""

from .ascii_gpu_data import (
    ASCII_CHAR_COUNT, ASCII_VEC_DIMS, ASCII_SHAPE_VECTORS,
    INTERNAL_SAMPLE_POS, EXTERNAL_SAMPLE_POS,
    AFFECTING_EXTERNAL, MAX_AFFECTING, MAX_PRESET_SIZE,
    CHARSET_CHOICES, CHARSET_PRESETS_DATA, CHARSET_PRESET_NAMES,
)
from .ascii_gpu_fonts_json import (FONT_NAMES,
                                    GLYPH_W, GLYPH_H, GLYPHS_PER_ROW,
                                    NUM_FONTS, CHARS_PER_FONT)


def _format_vec2_list(name: str, positions: list[tuple[float, float]]) -> str:
    """Format a GLSL const vec2 array from Python (x,y) pairs."""
    lines = [f"const vec2 {name}[{len(positions)}] = vec2[{len(positions)}]("]
    for idx, (x, y) in enumerate(positions):
        comma = "," if idx < len(positions) - 1 else ""
        lines.append(f"    vec2({x:.3f}, {y:.3f}){comma}")
    lines.append(");")
    return "\n".join(lines)


def _format_int_list(name: str, values: list[int], values_per_row: int = 12) -> str:
    """Format a GLSL const int array."""
    lines = [f"const int {name}[{len(values)}] = int[{len(values)}]("]
    for i in range(0, len(values), values_per_row):
        row = values[i:i + values_per_row]
        row_strs = [str(v) for v in row]
        comma = "," if i + values_per_row < len(values) else ""
        # Don't add comma if this is the last row
        if i + len(row) >= len(values):
            comma = ""
        lines.append("    " + ", ".join(row_strs) + comma)
    # If the last line ended with a comma (because values_per_row boundary
    # isn't the last element), we need to fix it.
    # Actually, let me just use a simpler approach
    lines = [f"const int {name}[{len(values)}] = int[{len(values)}]("]
    for idx, v in enumerate(values):
        comma = "," if idx < len(values) - 1 else ""
        lines.append(f"    {v}{comma}")
    lines.append(");")
    return "\n".join(lines)


def _format_float_list(name: str, values: list[float], dims: int) -> str:
    """Format a flat 1D GLSL const float array with dims-wide rows."""
    lines = [
        f"const float {name}[{len(values)}] = float[{len(values)}](",
    ]
    for idx in range(0, len(values), dims):
        row = values[idx:idx + dims]
        vals = ", ".join(f"{v:.6f}" for v in row)
        comma = "," if idx + dims < len(values) else ""
        lines.append(f"    {vals}{comma}")
    lines.append(");")
    return "\n".join(lines)


def _format_affecting_external() -> str:
    """Format the affecting-external map as a flat GLSL int array."""
    flat = []
    for row in AFFECTING_EXTERNAL:
        flat.extend(row + [-1] * (MAX_AFFECTING - len(row)))
    lines = [
        f"const int AFFECTING_EXTERNAL[{len(flat)}] = int[{len(flat)}](",
    ]
    for idx in range(0, len(flat), MAX_AFFECTING):
        row = flat[idx:idx + MAX_AFFECTING]
        vals = ", ".join(str(v) for v in row)
        comma = "," if idx + MAX_AFFECTING < len(flat) else ""
        lines.append(f"    {vals}{comma}")
    lines.append(");")
    return "\n".join(lines)


def build_ascii_glsl() -> str:
    """Build the complete ascii_art_gpu GLSL fragment shader source."""
    parts = []

    # ── Header: #define constants ──
    num_presets = len(CHARSET_PRESET_NAMES)
    nfonts = NUM_FONTS
    parts.append(f"""// ASCII GPU Art — shape-vector renderer
// 6D shape vectors for {ASCII_CHAR_COUNT} characters with staggered sampling
// Directional + global contrast enhancement for sharp edges
// {num_presets} character set presets, {nfonts} fonts

#define ASCII_CHAR_COUNT {ASCII_CHAR_COUNT}
#define ASCII_VEC_DIMS {ASCII_VEC_DIMS}
#define ATLAS_GLYPH_W {GLYPH_W}
#define ATLAS_GLYPH_H {GLYPH_H}
#define GLYPHS_PER_ROW {GLYPHS_PER_ROW}
#define NUM_FONTS {nfonts}
#define MAX_AFFECTING {MAX_AFFECTING}
#define CHARSET_COUNT {num_presets}
#define MAX_PRESET_SIZE {MAX_PRESET_SIZE}
""")

    # ── Sampling positions ──
    parts.append(_format_vec2_list("INTERNAL_SAMPLE_POS", INTERNAL_SAMPLE_POS))
    parts.append("")
    parts.append(_format_vec2_list("EXTERNAL_SAMPLE_POS", EXTERNAL_SAMPLE_POS))
    parts.append("")

    # ── Shape vectors (flat 1D) ──
    parts.append(_format_float_list("ASCII_SHAPE_VECTORS", ASCII_SHAPE_VECTORS, ASCII_VEC_DIMS))
    parts.append("")

    # ── Affecting external map ──
    parts.append(_format_affecting_external())
    parts.append("")

    # ── Glyph atlas sampler is a runtime uniform (loaded as u_glyph_atlas texture)
    parts.append("// Glyph atlas: uniform sampler2D u_glyph_atlas (loaded at render time)")
    parts.append(f"// Atlas: {GLYPH_W}×{GLYPH_H} per char, {GLYPHS_PER_ROW} per row, 6 fonts")
    parts.append("")

    # ── Character set presets ──
    parts.append(f"// {num_presets} presets, each padded to {MAX_PRESET_SIZE} entries (-1 = end sentinel)")
    parts.append(f"// Access: CHARSET_PRESETS[preset_idx * {MAX_PRESET_SIZE} + entry_idx]")
    parts.append(_format_int_list("CHARSET_PRESETS", CHARSET_PRESETS_DATA))
    parts.append("")

    # ── Main shader body ──
    parts.append("""uniform sampler2D u_glyph_atlas;  // glyph atlas texture
float _sample_lum(vec2 uv) {
    vec3 c = texture(u_texture, uv).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// Sample at a UV position with 5-point cross supersampling
float _sample_supersampled(vec2 uv, vec2 halfTexel) {
    float s = _sample_lum(uv);
    s += _sample_lum(uv + vec2(halfTexel.x, 0.0));
    s += _sample_lum(uv - vec2(halfTexel.x, 0.0));
    s += _sample_lum(uv + vec2(0.0, halfTexel.y));
    s += _sample_lum(uv - vec2(0.0, halfTexel.y));
    return s * 0.2;
}

void main() {
    float cellSize = max(u_cell_size, 4.0);
    // Monospace aspect: cells are ~2:1 height:width
    vec2 cSize = vec2(cellSize, cellSize * u_cell_aspect);
    vec2 cellOrigin = floor(gl_FragCoord.xy / cellSize) * cellSize;
    vec2 halfTexel = 0.5 / u_resolution;

    // ── 1. Compute 6D internal sampling vector ──
    float sv[ASCII_VEC_DIMS];
    for (int i = 0; i < ASCII_VEC_DIMS; i++) {
        vec2 uv = (cellOrigin + INTERNAL_SAMPLE_POS[i] * cSize) / u_resolution;
        sv[i] = _sample_supersampled(uv, halfTexel);
    }

    // ── 2. Compute 10D external sampling vector ──
    float ext[10];
    for (int i = 0; i < 10; i++) {
        vec2 uv = (cellOrigin + EXTERNAL_SAMPLE_POS[i] * cSize) / u_resolution;
        ext[i] = _sample_lum(uv);
    }

    // ── 3. Directional contrast enhancement ──
    if (u_directional_strength > 1.001) {
        for (int i = 0; i < ASCII_VEC_DIMS; i++) {
            float maxVal = sv[i];
            for (int j = 0; j < MAX_AFFECTING; j++) {
                int extIdx = AFFECTING_EXTERNAL[i * MAX_AFFECTING + j];
                if (extIdx < 0) break;
                maxVal = max(maxVal, ext[extIdx]);
            }
            if (maxVal > 0.001) {
                float n = sv[i] / maxVal;
                n = pow(n, u_directional_strength);
                sv[i] = n * maxVal;
            }
        }
    }

    // ── 4. Global contrast enhancement ──
    if (u_contrast > 1.001) {
        float maxVal = sv[0];
        for (int i = 1; i < ASCII_VEC_DIMS; i++) maxVal = max(maxVal, sv[i]);
        if (maxVal > 0.001) {
            for (int i = 0; i < ASCII_VEC_DIMS; i++) {
                float n = sv[i] / maxVal;
                n = pow(n, u_contrast);
                sv[i] = n * maxVal;
            }
        }
    }

    // ── 5. Apply gamma / invert to each component ──
    float gamma = max(u_gamma, 0.05);
    for (int i = 0; i < ASCII_VEC_DIMS; i++) {
        sv[i] = pow(clamp(sv[i], 0.0, 1.0), gamma);
        if (u_invert == 1) sv[i] = 1.0 - sv[i];
    }

    // ── 6. Nearest-neighbor search in charset-preset 6D shape space ──
    int presetBase = u_charset * MAX_PRESET_SIZE;
    float bestDist = 1e10;
    int bestIdx = 0;
    for (int j = 0; j < MAX_PRESET_SIZE; j++) {
        int ci = CHARSET_PRESETS[presetBase + j];
        if (ci < 0) break; // -1 sentinel: end of this preset
        float dist = 0.0;
        for (int di = 0; di < ASCII_VEC_DIMS; di++) {
            float diff = sv[di] - ASCII_SHAPE_VECTORS[ci * ASCII_VEC_DIMS + di];
            dist += diff * diff;
        }
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = ci;
        }
    }

    // ── 7. Render glyph from atlas texture ──
    vec2 p = (gl_FragCoord.xy - cellOrigin) / cellSize;
    int atlasCol = bestIdx % GLYPHS_PER_ROW;
    int atlasRow = bestIdx / GLYPHS_PER_ROW;
    vec2 atlasPixelOrigin = vec2(float(atlasCol) * float(ATLAS_GLYPH_W),
                                 float(atlasRow) * float(ATLAS_GLYPH_H));
    vec2 atlasPixel = atlasPixelOrigin + p * vec2(float(ATLAS_GLYPH_W), float(ATLAS_GLYPH_H));
    vec2 atlasSize = vec2(textureSize(u_glyph_atlas, 0));
    vec2 atlasUV = atlasPixel / atlasSize;
    float glyphVal = texture(u_glyph_atlas, atlasUV).r;
    float isLit = smoothstep(0.3, 0.6, glyphVal);

    // ── 8. Coloring ──
    vec3 src = texture(u_texture, (cellOrigin + 0.5 * cellSize) / u_resolution).rgb;
    vec3 col;
    if (u_mode == 1)      col = mix(u_bg_color, src, isLit);                      // colored
    else if (u_mode == 2) {
        float g = dot(src, vec3(0.299, 0.587, 0.114));
        col = mix(vec3(0.0, 0.05, 0.0), vec3(0.2, 1.0, 0.3) * (0.4 + 0.6 * g), isLit); // terminal
    } else                col = mix(u_bg_color, u_fg_color, isLit);               // mono
    f_color = vec4(col, 1.0);
}
""")

    return "\n".join(parts)
