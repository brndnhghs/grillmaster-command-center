"""julia — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("julia", 'Julia set fractal (client-GPU twin of node 66)', "procedural",
          '''
void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
    vec2 c = vec2(-0.7269, 0.1889);  // node 66's famous default constant (string param unmapped)
    vec2 z = uv * 3.0;              // fixed full view (node 66 has no zoom param)
    int n = 0;
    float last2 = 0.0;
    const float MAXI = 500.0;
    for (int i = 0; i < 500; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        last2 = dot(z, z);
        if (last2 > u_escape_radius * u_escape_radius || n >= u_iterations) break;
        n++;
    }
    float t = (n >= u_iterations - 0.5) ? 0.0
            : clamp((n + 1.0 - log(max(log(last2)*0.5, 1.0001))/log(2.0)) / u_iterations, 0.0, 1.0);
    f_color = vec4(0.5 + 0.5 * cos(t * 6.28318 + vec3(0.0, 2.0, 4.0)), 1.0);
}
''',
          uniforms={
  "iterations": {
    "glsl": "float",
    "min": 30.0,
    "max": 500.0,
    "default": 100,
    "description": "max iterations"
  },
  "escape_radius": {
    "glsl": "float",
    "min": 1.5,
    "max": 10.0,
    "default": 2.0,
    "description": "escape radius"
  }
})