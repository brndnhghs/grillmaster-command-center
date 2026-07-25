"""GPU shader nodes. Auto-imports every sibling module so registration runs."""
import importlib
import os
import pkgutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _finder, _name, _ispkg in pkgutil.iter_modules([_THIS_DIR]):
    importlib.import_module(f"{__name__}.{_name}")

# Re-export names used by server.py and test files
from ._shared import (
    GPU_SHADER_NODE_MAP,
    CLIENT_GPU_SHIMS,
    CLIENT_GPU_SIMS,
    _PROC_SHADERS,
    _FILT_SHADERS,
    _TYPED_SHADER_NODES,
    SHADER_NAMES,
    GPU_PREVIEW_DROP_ALLOW,
    is_param_justified_drop,
)
