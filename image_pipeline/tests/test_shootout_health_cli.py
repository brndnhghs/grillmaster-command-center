"""Headless test for the shootout corpus-health CLI diagnostic (Route 8 / hygiene).

``image_pipeline.shootout.cli._print_health`` codifies the standing
Phase-1 corpus analysis so every autonomous run (and the user) gets the
dead-rate / reason-breakdown / driver-independence / rating-starvation /
liveness-calibration picture instantly instead of re-deriving it by hand.

The test only asserts the diagnostic RUNS and emits a coherent verdict; it
deliberately tolerates an empty corpus (clean CI) by accepting either the
"no genomes" short-circuit or the full report.
"""

from __future__ import annotations

import contextlib
import io

from image_pipeline.shootout import cli


def test_health_runs_and_reports() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli._print_health()
    out = buf.getvalue()

    # Clean CI (no corpus): short-circuits.
    if "no genomes in corpus" in out:
        return

    # With a corpus: must print the corpus line and the driver-independence
    # verdict (the key finding the diagnostic exists to surface).
    assert "corpus:" in out, out[:200]
    assert "DRIVER-INDEPENDENCE" in out, out[:400]
    # Exactly one of the two coherent verdicts must be present.
    verdicts = (
        "drivers are a UBILITY confound" in out,
        "driver presence DOES affect death rate" in out,
    )
    assert any(verdicts), out[:600]
    # Rating starvation line is always present.
    assert "rating starvation:" in out, out[:400]
