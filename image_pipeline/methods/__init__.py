"""Auto-import all method group modules. Each module adds its methods to the registry."""
from . import fractals, simulations, filters, patterns, cli_tools, math_art, ml_models, gpu_shaders
from . import simulations_cellular  # #58 Cellular Automata (Variants) — toroidal, Brian's Brain, age coloring
from . import codegen  # package with individual method files
from . import compositing  # blend, math_merge, field_combine, particle_merge, apply_mask