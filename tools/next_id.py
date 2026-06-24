#!/usr/bin/env python3
"""
Print the next available method ID.
Usage: uv run python tools/next_id.py
       uv run python tools/next_id.py --reserve 5
"""
import ast, sys
from pathlib import Path

def get_used_ids() -> set[int]:
    ids = set()
    for f in Path("image_pipeline/methods").rglob("*.py"):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in getattr(node, 'keywords', []):
                    if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                        ids.add(int(kw.value.value))
    return ids

def main():
    reserve = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == '--reserve' else 1
    used = get_used_ids()
    next_id = max(used) + 1 if used else 1
    if reserve == 1:
        print(f"Next available method ID: {next_id}")
    else:
        ids = list(range(next_id, next_id + reserve))
        print(f"Next {reserve} available IDs: {ids}")

if __name__ == "__main__":
    main()
