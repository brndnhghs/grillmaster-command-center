"""kpz_display — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("kpz_display",
          "KPZ display: hillshaded height -> terrain grayscale (matches _render_terrain)",
          "procedural", '''
void main() {
    vec2 texel = 1.0 / u_resolution;
    float h = texture(u_texture, v_uv).r;
    float dx = (texture(u_texture, v_uv + vec2(texel.x,0.0)).r
                - texture(u_texture, v_uv + vec2(-texel.x,0.0)).r) * 0.5;
    float dy = (texture(u_texture, v_uv + vec2(0.0,texel.y)).r
                - texture(u_texture, v_uv + vec2(0.0,-texel.y)).r) * 0.5;
    vec3 nrm = normalize(vec3(-dx, -dy, 0.08));
    vec3 sun = normalize(vec3(0.6, 0.6, 0.7));
    float shade = clamp(dot(nrm, sun), 0.0, 1.0);
    float t = clamp(h * 0.25 + 0.5, 0.0, 1.0);
    float g = clamp(shade * 0.7 + t * 0.4, 0.0, 1.0);
    f_color = vec4(vec3(g), 1.0);
}
''')