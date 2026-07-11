"""Taste model v1 — ridge regressor over genome features (plan §9).

Small data, refit-cheap, zero new dependencies (numpy closed form). v1 is
log + train only: it reports a held-out metric so you can watch it learn,
but does NOT gate what is shown (decision #6). Phase 2 will use predict()
to bias generation.
"""
from __future__ import annotations

import time

import numpy as np

from . import store

MIN_SAMPLES = 8
RIDGE_LAMBDA = 1.0


def _vectorize(records: list[dict], feature_names: list[str]) -> np.ndarray:
    X = np.zeros((len(records), len(feature_names)), dtype=np.float64)
    idx = {k: i for i, k in enumerate(feature_names)}
    for r, rec in enumerate(records):
        for k, v in (rec.get("features") or {}).items():
            j = idx.get(k)
            if j is not None and isinstance(v, (int, float)):
                X[r, j] = v
    return X


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Standardized ridge with intercept; returns (coef in raw space precursor)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    A = Xb.T @ Xb + lam * np.eye(d + 1)
    A[-1, -1] -= lam  # don't penalize the intercept
    w = np.linalg.solve(A, Xb.T @ y)
    return w[:-1], float(w[-1])


def train(records: list[dict] | None = None,
          lam: float = RIDGE_LAMBDA) -> dict:
    """Fit on the ratings dataset; k-fold held-out metrics vs mean baseline.
    Returns the model artifact (also what /api/shootout/train reports)."""
    if records is None:
        records = store.load_ratings()
    records = [r for r in records
               if isinstance(r.get("rating"), (int, float)) and r.get("features")]
    n = len(records)
    if n < MIN_SAMPLES:
        return {"trained": False, "n_samples": n,
                "note": f"need >= {MIN_SAMPLES} ratings"}

    feature_names = sorted({k for r in records for k in r["features"]})
    X_raw = _vectorize(records, feature_names)
    y = np.array([float(r["rating"]) for r in records])

    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X = (X_raw - mu) / sd

    # ── k-fold held-out metric ────────────────────────────────────
    k = min(5, n)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    preds = np.zeros(n)
    base = np.zeros(n)
    for fold in range(k):
        test = perm[fold::k]
        trn = np.setdiff1d(perm, test)
        coef, b = _fit_ridge(X[trn], y[trn], lam)
        preds[test] = X[test] @ coef + b
        base[test] = y[trn].mean()
    preds = preds.clip(1, 5)
    mae = float(np.abs(preds - y).mean())
    baseline_mae = float(np.abs(base - y).mean())
    if y.std() > 1e-9 and preds.std() > 1e-9:
        corr = float(np.corrcoef(preds, y)[0, 1])
    else:
        corr = 0.0

    # ── Final fit on everything ───────────────────────────────────
    coef, intercept = _fit_ridge(X, y, lam)

    artifact = {
        "trained": True,
        "model": "ridge",
        "lambda": lam,
        "n_samples": n,
        "feature_names": feature_names,
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "metrics": {
            "cv_mae": round(mae, 4),
            "baseline_mae": round(baseline_mae, 4),
            "cv_corr": round(corr, 4),
            "beats_baseline": mae < baseline_mae,
        },
        "trained_at": time.time(),
    }
    store.save_model(artifact)
    return artifact


def predict(features: dict, artifact: dict | None = None) -> float | None:
    """Predicted star rating for a feature dict; None when no model exists."""
    if artifact is None:
        artifact = store.load_model()
    if not artifact or not artifact.get("trained"):
        return None
    names = artifact["feature_names"]
    x = np.array([float(features.get(k, 0.0)) for k in names])
    x = (x - np.array(artifact["mean"])) / np.array(artifact["std"])
    val = float(x @ np.array(artifact["coef"]) + artifact["intercept"])
    return min(max(val, 1.0), 5.0)
