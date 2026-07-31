"""swirl_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Typed-uniform nodes 232-237 (categorical coverage expansion, 2026-07-10) ──
# swirl displacement, chromatic aberration, halftone, concentric rings,
# truchet tiles, pixelate/mosaic. Same typed-uniform contract as 226-231:
# every variable is a real node param + wireable SCALAR port; filters take
# image_in: IMAGE. Bodies stay in the GL330/ES300 parity subset.

_register("swirl_gpu", "Swirl / vortex displacement of the input (typed)",
          "filter", '''
void main() {
    vec2 uv = v_uv - 0.5;
    float r = length(uv);
    float amt = (u_strength) * smoothstep(u_radius, 0.0, r);
    float a = amt + u_time * u_spin;
    uv = rot(a) * uv;
    vec3 src = texture(u_texture, fract(uv + 0.5)).rgb;
    f_color = vec4(src, 1.0);
}
''', uniforms={
    "strength": {"glsl": "float", "min": -6.0, "max": 6.0, "default": 3.0,
                 "description": "swirl strength (signed)"},
    "radius":   {"glsl": "float", "min": 0.1, "max": 1.2, "default": 0.6,
                 "description": "swirl falloff radius"},
    "spin":     {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.0,
                 "description": "animated spin speed"},
})