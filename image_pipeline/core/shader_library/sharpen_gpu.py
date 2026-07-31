"""sharpen_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("sharpen_gpu", "Unsharp-mask sharpen of the input (typed)",
          "filter", '''
void main() {
    vec2 px = u_radius / u_resolution;
    vec3 c  = texture(u_texture, v_uv).rgb;
    vec3 nb = texture(u_texture, v_uv + px * vec2( 0.0,  1.0)).rgb
            + texture(u_texture, v_uv + px * vec2( 0.0, -1.0)).rgb
            + texture(u_texture, v_uv + px * vec2( 1.0,  0.0)).rgb
            + texture(u_texture, v_uv + px * vec2(-1.0,  0.0)).rgb;
    vec3 blur = nb * 0.25;
    vec3 sharp = c + (c - blur) * u_strength;
    f_color = vec4(clamp(sharp, 0.0, 1.0), 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": 0.0, "max": 5.0, "default": 1.5,
                 "description": "sharpen strength"},
    "radius":   {"glsl": "float", "min": 0.5, "max": 6.0, "default": 1.0,
                 "description": "sample radius (px)"},
})