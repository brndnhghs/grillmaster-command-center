# Module: `core/node_tester.py`

## Purpose
Run every registered method in isolation and report results. Supports batch-apply of Node Doctor fixes.

## Responsibilities
- Test each method with default params (no graph wiring)
- Test each method with edge-case param values
- Structured pass/fail reporting with error traces and output stats
- Integration with Node Doctor for batch fixes

## Public Interfaces
- `TestResult` — single method test result
- `TestReport` — aggregate report for full test run

## Dependencies
- `registry.py` — iterates all methods
- `numpy`

## Consumers
- `server.py` — Node Tester API endpoint
- `tools/audit_methods.py` — pre-commit gate (indirect)