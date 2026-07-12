"""Registration regression guard for the pkgutil auto-discovery refactor.

After image_pipeline/methods/*/__init__.py switched from explicit
`from . import x` lists to `pkgutil.iter_modules` auto-discovery, every
`.py` method module under a category subpackage is imported automatically and
its `@method` nodes register. This test makes that contract *observable*: if a
future change ever drops a module from registration (import error swallowed, a
file added outside a scanned package, a typo in an id), the test fails instead
of silently shipping a missing node.

It is intentionally dependency-free: it parses each source file for `@method(id=...)`
literals and asserts those ids appear in the live in-process registry. No HTTP,
no network, no GPU.
"""
import os
import re

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_METHODS_DIR = os.path.join(_REPO, "image_pipeline", "methods")

# Subpackages that hold @method nodes (mirrors the auto-discovery roots).
_CATEGORY_PKGS = [
    "codegen", "compositing", "filters", "fractals", "math_art",
    "patterns", "simulations", "system", "cli_tools", "ml_models",
    "gpu_shaders", "p5_sketches", "channels", "simulations_cellular",
    "custom_shader", "io_nodes", "blender_render",
]

# Files we never expect to define @method nodes.
_SKIP_FILES = {"__init__.py"}


def _expected_ids():
    """Return {(cat, file): [ids]} for every method module that declares @method."""
    found = {}
    for cat in _CATEGORY_PKGS:
        cat_dir = os.path.join(_METHODS_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fn in sorted(os.listdir(cat_dir)):
            if not fn.endswith(".py") or fn in _SKIP_FILES or fn.startswith("_"):
                continue
            path = os.path.join(cat_dir, fn)
            try:
                text = open(path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            ids = []
            for m in re.finditer(r"@method\s*\((.*?)\)\s*\n\s*def", text, re.DOTALL):
                block = m.group(1)
                idm = re.search(r"id\s*=\s*[\"']?(\d+)", block)
                if idm:
                    ids.append(idm.group(1))
            if ids:
                found[(cat, fn)] = ids
    return found


@pytest.fixture(scope="module")
def registered():
    # Importing the package triggers @method registration (same path the server uses).
    import image_pipeline.methods  # noqa: F401
    from image_pipeline.server import get_node_defs
    return set(get_node_defs().keys())


def test_all_method_modules_registered(registered):
    expected = _expected_ids()
    assert expected, "no @method nodes discovered — scanner misconfigured?"
    missing = []
    for (cat, fn), ids in expected.items():
        for nid in ids:
            if nid not in registered:
                missing.append(f"{cat}/{fn}:#{nid}")
    assert not missing, (
        f"{len(missing)} declared @method node(s) are NOT registered:\n"
        + "\n".join(f"  - {m}" for m in missing)
    )


def test_registration_count_healthy(registered):
    # Sanity floor: the pipeline ships hundreds of nodes. A catastrophic
    # registration loss (e.g. auto-discovery loop breaking) would drop this.
    assert len(registered) >= 300, f"only {len(registered)} nodes registered"
