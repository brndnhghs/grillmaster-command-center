"""color_grade_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_typed



# 493 Color Grading (OKLab) — perceptually-uniform brightness/contrast/gamma/
# saturation/hue/white-balance grade in the OKLab space (Ottosson 2020). Every
# op happens on the wired IMAGE in OKLab so saturation changes do not shift hue
# and contrast changes are perceptually even. Mirrors the CPU node 493 math
# (see methods/filters/color_grade.py) as a typed-uniform GPU filter twin. The
# CPU fn stays authoritative for the synthetic `source` generators + all string
# choice params; string params (source/palette/invert/anim_mode) are unmapped.
_register("color_grade_gpu", "GPU color grading in OKLab perceptual space", "filter", _filter_typed('''
    // sRGB -> linear (avoid GLSL step() — `step` is a reserved vec2 here)
    vec3 c = orig.rgb;
    vec3 lin = vec3(
        c.r <= 0.04045 ? c.r / 12.92 : pow((c.r + 0.055) / 1.055, 2.4),
        c.g <= 0.04045 ? c.g / 12.92 : pow((c.g + 0.055) / 1.055, 2.4),
        c.b <= 0.04045 ? c.b / 12.92 : pow((c.b + 0.055) / 1.055, 2.4));
    // linear RGB -> OKLab
    float l = 0.4122214708*lin.r + 0.5363325363*lin.g + 0.0514459929*lin.b;
    float m = 0.2119034982*lin.r + 0.6806995451*lin.g + 0.1073969566*lin.b;
    float s = 0.0883024619*lin.r + 0.2817188376*lin.g + 0.6299787005*lin.b;
    float l_ = sign(l)*pow(abs(l), 1.0/3.0);
    float m_ = sign(m)*pow(abs(m), 1.0/3.0);
    float s_ = sign(s)*pow(abs(s), 1.0/3.0);
    float L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_;
    float A = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_;
    float B = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_;
    // Lightness grade: exposure, contrast (pivot 0.5), gamma
    L = L * pow(2.0, u_exposure);
    L = (L - 0.5) * u_contrast + 0.5;
    L = sign(L) * pow(abs(L), 1.0 / max(0.05, u_gamma));
    // Chroma grade: hue rotate + saturation
    float hr = radians(u_hue_rotate);
    float ca = cos(hr), sa = sin(hr);
    vec2 ab = vec2(A * ca - B * sa, A * sa + B * ca) * u_saturation;
    // White balance: temperature (a warm/cool ~ +b/-b via a & b) + tint
    ab.x += u_tint * 0.10;
    ab.y += u_temperature * 0.10;
    A = ab.x; B = ab.y;
    // Radial vignette on lightness
    vec2 vp = uv - 0.5;
    float vr = length(vp) * 1.41421356;
    L *= mix(1.0, 1.0 - vr * vr, clamp(u_vignette, 0.0, 1.0));
    // OKLab -> linear RGB
    float li = L + 0.3963377774*A + 0.2158037573*B;
    float mi = L - 0.1055613458*A - 0.0638541728*B;
    float si = L - 0.0894841775*A - 1.2914855480*B;
    li = li*li*li; mi = mi*mi*mi; si = si*si*si;
    vec3 rgb = vec3(
         4.0767416621*li - 3.3077115913*mi + 0.2309699292*si,
        -1.2684380046*li + 2.6097574011*mi - 0.3413193965*si,
        -0.0041960863*li - 0.7034186147*mi + 1.7076147010*si);
    rgb = clamp(rgb, 0.0, 1.0);
    // linear -> sRGB (avoid step())
    vec3 outc = vec3(
        rgb.r <= 0.0031308 ? rgb.r * 12.92 : 1.055 * pow(rgb.r, 1.0/2.4) - 0.055,
        rgb.g <= 0.0031308 ? rgb.g * 12.92 : 1.055 * pow(rgb.g, 1.0/2.4) - 0.055,
        rgb.b <= 0.0031308 ? rgb.b * 12.92 : 1.055 * pow(rgb.b, 1.0/2.4) - 0.055);
    f_color = vec4(clamp(outc, 0.0, 1.0), orig.a);
'''), uniforms={
    "exposure": {"glsl": "float", "min": -3.0, "max": 3.0, "default": 0.0, "description": "exposure stops (2^ev)"},
    "contrast": {"glsl": "float", "min": 0.2, "max": 3.0, "default": 1.0, "description": "lightness contrast (pivot 0.5)"},
    "gamma": {"glsl": "float", "min": 0.3, "max": 3.0, "default": 1.0, "description": "lightness gamma"},
    "saturation": {"glsl": "float", "min": 0.0, "max": 3.0, "default": 1.0, "description": "OKLab chroma scale"},
    "hue_rotate": {"glsl": "float", "min": -180.0, "max": 180.0, "default": 0.0, "description": "hue rotation (deg)"},
    "temperature": {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "warm/cool white balance"},
    "tint": {"glsl": "float", "min": -1.0, "max": 1.0, "default": 0.0, "description": "green/magenta tint"},
    "vignette": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.0, "description": "radial darkening"},
})