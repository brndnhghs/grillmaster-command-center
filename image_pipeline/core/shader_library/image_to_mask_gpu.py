"""image_to_mask_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _filter_shader



_register("image_to_mask_gpu",
          "Luminance/channel mask extraction (client-GPU twin of __image_to_mask__)",
          "filter", _filter_shader('''
    // Client-GPU live-preview twin of the Image to Mask compositing node.
    // The node's `mode` is a STRING choice param (luminance/red/green/blue/
    // alpha_from_white/invert_luminance) and per GPU pitfall #14 string params
    // are NOT mapped to numeric uniforms — so this twin renders the DEFAULT
    // `luminance` mode. The CPU fn stays authoritative for all six modes on
    // export. Output is a grayscale mask preview (mask replicated to RGB).
    float lum = dot(orig.rgb, vec3(0.299, 0.587, 0.114));
    float m = clamp(lum, 0.0, 1.0);
    f_color = vec4(vec3(m), 1.0);
'''))