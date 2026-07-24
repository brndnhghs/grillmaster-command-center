# Module: `core/expr.py`

## Purpose
Safe per-frame expression evaluator for node graph params. Allows users to write math expressions as param values that are evaluated each frame with `frame`, `seed`, and `t` in scope.

## Responsibilities
- Parse and safety-check expression strings using AST
- Maintain a compiled-code cache for performance
- Evaluate expressions with a whitelisted function/variable namespace
- Fall back to numeric parsing for plain number strings

## Public Interfaces

### `eval_param(value, frame, seed, total_frames=100) -> Any`
Evaluate a param value, expanding expression strings per frame.

- Non-string values returned unchanged
- Plain numeric strings → fast path `float(value)`
- Expression strings → compile + eval with safe namespace
- On any error → return 0.0

### Available in expressions
**Variables:** `frame`, `seed`, `t` (= `frame / total_frames`)
**Functions:** `sin`, `cos`, `tan`, `abs`, `floor`, `ceil`, `round`, `sqrt`, `log`, `pow`, `min`, `max`
**Constants:** `pi`, `e`
**Pseudo-noise:** `noise(x)` = `sin(x * 127.1 + 311.7) * 0.5 + 0.5`

### Safety
- Only whitelisted AST nodes allowed (binops, unaryops, calls to safe names, constants, names)
- Calls only to `_SAFE_CALL_NAMES` — no arbitrary function calls
- Variables only in `_SAFE_NAMES | {"frame", "seed", "t"}`
- Executed with `{"__builtins__": {}}` — no builtins access

## Internal

### `_compile_expr(expr) -> code object or None`
Parse, safety-check, and compile expression. Results cached in `_COMPILED_CACHE` (max 1024 entries).

### Safety Check: `_is_safe(node) -> bool`
Walks AST recursively, rejecting any disallowed node types or unsafe names.

## Dependencies
- stdlib: `ast`, `math`

## Consumers
- `core/graph.py` — `_make_node_def()` and executor call `eval_param` on expression-typed param values

## Performance
- Parsing + safety + compile: ~50-200µs
- Compiled cache: 1024 entry LRU-style (clears when full)
- Plain numeric strings: fast path avoids AST entirely