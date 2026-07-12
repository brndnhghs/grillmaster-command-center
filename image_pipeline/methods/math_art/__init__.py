"""Math-art methods (Ulam, Maze, Circle Packing, etc.). Auto-imports every sibling module so new methods register automatically."""
import importlib
import os
import pkgutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _finder, _name, _ispkg in pkgutil.iter_modules([_THIS_DIR]):
    importlib.import_module(f"{__name__}.{_name}")
