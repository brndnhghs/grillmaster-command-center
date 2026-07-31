"""dither_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dither_gpu", "Ordered (Bayer 4x4) dither of the input (typed)",
          "filter", '''
void main() {
    vec3 src = texture(u_texture, v_uv).rgb;
    vec2 gp = mod(floor(v_uv * u_resolution / max(u_scale, 1.0)), 4.0);
    int ix = int(gp.x); int iy = int(gp.y);
    float bayer[16];
    bayer[0]=0.0;  bayer[1]=8.0;  bayer[2]=2.0;  bayer[3]=10.0;
    bayer[4]=12.0; bayer[5]=4.0;  bayer[6]=14.0; bayer[7]=6.0;
    bayer[8]=3.0;  bayer[9]=11.0; bayer[10]=1.0; bayer[11]=9.0;
    bayer[12]=15.0;bayer[13]=7.0; bayer[14]=13.0;bayer[15]=5.0;
    int bi = iy * 4 + ix;
    float thr = (bayer[bi] + 0.5) / 16.0;
    float lv = max(float(u_levels), 2.0);
    vec3 dithered = floor(src * lv + (thr - 0.5)) / (lv - 1.0);
    dithered = clamp(dithered, 0.0, 1.0);
    f_color = vec4(mix(src, dithered, clamp(u_amount, 0.0, 1.0)), 1.0);
}
''', uniforms={
    "levels": {"glsl": "int", "min": 2, "max": 16, "default": 3,
               "description": "output levels per channel"},
    "scale":  {"glsl": "float", "min": 1.0, "max": 8.0, "default": 1.0,
               "description": "dither pattern scale (px)"},
    "amount": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 1.0,
               "description": "effect amount"},
})