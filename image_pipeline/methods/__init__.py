"""Auto-register all method group modules. Each module adds its methods to the registry. Sub-packages auto-discover their own modules (new method files register automatically)."""
import importlib
import os
import pkgutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _finder, _name, _ispkg in pkgutil.iter_modules([_THIS_DIR]):
    importlib.import_module(f"{__name__}.{_name}")
