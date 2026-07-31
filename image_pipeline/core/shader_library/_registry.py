"""Shared shader registry — the single SHADERS dict + _register().

Split out of core/shaders.py so shader_library modules can register without
importing core.shaders (no circular imports): each module in this package
calls ``_register(...)`` at import time and core.shaders imports the same
dict object from here.
"""
from __future__ import annotations

SHADERS: dict = {}


def _register(name: str, description: str, shader_type: str, source: str,
              uniforms: dict | None = None):
    """Register a shader in the SHADERS dict."""
    SHADERS[name] = {
        "name": name,
        "description": description,
        "type": shader_type,
        "source": source,
        "uniforms": uniforms or {},
    }
