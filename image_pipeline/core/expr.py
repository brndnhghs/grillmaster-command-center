"""Safe per-frame expression evaluator for node graph params."""
from __future__ import annotations

import ast
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Safe names available in expression context
_SAFE_NAMES: dict[str, Any] = {
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "pi":    math.pi,
    "e":     math.e,
    "abs":   abs,
    "floor": math.floor,
    "ceil":  math.ceil,
    "round": round,
    "sqrt":  math.sqrt,
    "log":   math.log,
    "pow":   math.pow,
    "min":   min,
    "max":   max,
    # Simple deterministic pseudo-noise (not seeded, just a smooth bumpy function)
    "noise": lambda x: math.sin(x * 127.1 + 311.7) * 0.5 + 0.5,
}

# Allowed AST node types for safe eval
_ALLOWED = (
    ast.Expression,
    ast.Constant,
    ast.BinOp, ast.UnaryOp,
    ast.Call, ast.Name,
    ast.IfExp, ast.Compare, ast.BoolOp,
    ast.Load,   # name load context — child of every ast.Name
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not,
)

_SAFE_CALL_NAMES = frozenset(_SAFE_NAMES.keys())


def _is_safe(node: ast.AST) -> bool:
    if not isinstance(node, _ALLOWED):
        return False
    # Only allow calls to whitelisted names (no arbitrary callables)
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _SAFE_CALL_NAMES):
            return False
    # Name references must be whitelisted variables (frame, seed, t) or math names
    if isinstance(node, ast.Name) and node.id not in (_SAFE_CALL_NAMES | {"frame", "seed", "t"}):
        return False
    return all(_is_safe(child) for child in ast.iter_child_nodes(node))


# Compiled-code cache: expression string → code object, or None for strings
# that failed parsing/safety (so bad strings don't re-parse every frame).
# Parsing + safety-walking + compiling an expression cost ~50-200µs; in live
# mode every expression param re-ran that per node per frame.
_COMPILED_CACHE: dict[str, Any] = {}
_COMPILED_CACHE_MAX = 1024


def _compile_expr(expr: str):
    """Parse, safety-check, and compile an expression. Cached; None = rejected."""
    if expr in _COMPILED_CACHE:
        return _COMPILED_CACHE[expr]

    code = None
    try:
        tree = ast.parse(expr, mode="eval")
        if _is_safe(tree):
            code = compile(tree, "<expr>", "eval")
        else:
            logger.warning("expr: unsafe expression rejected: %r", expr)
    except SyntaxError as exc:
        logger.warning("expr: syntax error in %r: %s", expr, exc)

    if len(_COMPILED_CACHE) >= _COMPILED_CACHE_MAX:
        _COMPILED_CACHE.clear()
    _COMPILED_CACHE[expr] = code
    return code


def eval_param(value: Any, frame: int, seed: int, total_frames: int = 100) -> Any:
    """Evaluate a param value, expanding expression strings per frame.

    Non-string values are returned unchanged. String values are parsed as safe
    math expressions with variables: frame, seed, t (=frame/total_frames),
    plus standard math functions. Falls back to float(value) then 0.0 on error.
    """
    if not isinstance(value, str):
        return value

    expr = value.strip()
    if not expr:
        return 0.0

    # Plain numeric string — fast path, no AST needed
    try:
        return float(expr)
    except (ValueError, TypeError):
        pass

    code = _compile_expr(expr)
    if code is None:
        return 0.0

    t = frame / max(total_frames, 1)
    ctx = {**_SAFE_NAMES, "frame": frame, "seed": seed, "t": t}

    try:
        result = eval(code, {"__builtins__": {}}, ctx)  # noqa: S307
        return float(result)
    except Exception as exc:
        logger.warning("expr: eval failed for %r: %s", expr, exc)
        return 0.0
