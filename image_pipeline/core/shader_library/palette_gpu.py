"""palette_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("palette_gpu",
          "Color palette swatches (client-GPU twin of node 10)",
          "procedural", '''
void main() {
    // u_params.x = n_colors (2..32), u_params.y = saturation (0.5=auto),
    // u_params.z = hue_offset (0..1), u_params.w = value (0.5=auto).
    float ncols = floor(2.0 + u_hue_offset * 30.0);
    float hueOff = u_value;
    float sat = (u_saturation <= 0.0) ? 0.75 : clamp(u_saturation, 0.0, 1.0);
    float val = 0.95;

    // Arrange the hue ramp as a vertical band of swatches across the canvas.
    int col = int(floor(v_uv.x * ncols));
    float fn = (ncols > 0.5) ? (float(col) / ncols) : 0.0;
    float hue = fract(hueOff + fn);
    // vertical brightness variation inside each swatch so it reads as a palette.
    float band = step(0.08, v_uv.y) * step(v_uv.y, 0.92);
    float v = val * (0.55 + 0.45 * v_uv.y);
    vec3 col3 = clamp(vec3(
        abs(fract(hue + 1.0/3.0) * 2.0 - 1.0),
        abs(fract(hue) * 2.0 - 1.0),
        abs(fract(hue - 1.0/3.0) * 2.0 - 1.0)
    ), 0.0, 1.0);
    col3 = mix(vec3(dot(col3, vec3(0.299,0.587,0.114))), col3, sat) * v;
    f_color = vec4(mix(vec3(0.05), col3, band), 1.0);
}
''',
    uniforms={
    "hue_offset": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "hue offset"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "saturation"},
    "value": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "value (kept for parity)"}
}
    )