"""GPU shader registrations — one module per shader, loaded dynamically.

Each module in this package (except _registry/_helpers) calls ``_register(...)``
at import time, populating ``core.shaders.SHADERS``. core/shaders.py calls
``load_all()`` at the end of its import, so the dict is fully populated before
any render code runs. Adding a shader = adding one file here.
"""
import importlib
import pkgutil

_LOADED = False


def load_all():
    """Import every shader module in this package (idempotent)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue  # _registry, _helpers
        importlib.import_module(f"{__name__}.{mod.name}")
