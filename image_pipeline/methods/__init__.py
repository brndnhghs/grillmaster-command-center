"""Auto-import all method group modules. Each module adds its methods to the registry."""
from . import fractals, simulations, filters, patterns, cli_tools, math_art, ml_models, gpu_shaders
from . import p5_sketches  # p5.js headless browser sketches
from . import channels  # CHOP-like generators: Counter, Ramp, LFO, Beats, Noise1D, Envelope, Math, Logic, Blend
from . import simulations_cellular  # #58 Cellular Automata (Variants) — toroidal, Brian's Brain, age coloring
from . import codegen  # package with individual method files
from . import compositing  # blend, math_merge, field_combine, particle_merge, apply_mask
from . import system  # Timeline — global animation clock node
from . import custom_shader  # Custom GLSL Shader — live GLSL editor node
from . import io_nodes  # Image Import / Video Import — file source nodes
from . import blender_render  # Blender Render — live Blender MCP 3D source node