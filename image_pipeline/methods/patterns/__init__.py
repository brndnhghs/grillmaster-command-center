"""Pattern methods — Truchet, Quasicrystal, Moiré, Worley, Wallpaper, XDoG, etc.
Auto-imports every sibling module so new methods register automatically."""
import importlib
import pkgutil

for _finder, _name, _ispkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_name}")
