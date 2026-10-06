"""Metrics for the binary task relevant (1) vs irrelevant (0)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {"tp": int(((y_true == 1) & (y_pred == 1)).sum()), "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()), "tn": int(((y_true == 0) & (y_pred == 0)).sum())}


def metrics(y_true, y_pred, weights=None) -> dict:
    """Precision / recall / F1 for the Relevant class, accuracy, confusion counts (optionally weighted)."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    w = np.ones(len(y_true)) if weights is None else np.asarray(weights, dtype=float)
    tp = w[(y_true == 1) & (y_pred == 1)].sum()
    fp = w[(y_true == 0) & (y_pred == 1)].sum()
    fn = w[(y_true == 1) & (y_pred == 0)].sum()
    tn = w[(y_true == 0) & (y_pred == 0)].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tp + tn) / w.sum() if w.sum() else 0.0, "n": int(len(y_true))}
    if weights is None:
        out.update(confusion(y_true, y_pred))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    return round(float(roc_auc_score(y_true, scores)), 4) if 0 < y_true.sum() < len(y_true) else float("nan")


def bootstrap_ci(y_true, y_pred, key: str = "f1", n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(metrics(y_true[idx], y_pred[idx])[key])
    return round(float(np.quantile(vals, alpha / 2)), 4), round(float(np.quantile(vals, 1 - alpha / 2)), 4)


def best_threshold(y_true, proba, beta: float = 2.0, lo: float = 0.2, hi: float = 0.8) -> float:
    """Threshold that maximises F-beta (beta=2 favours recall of Relevant); searched on out-of-fold dev scores only."""
    y_true = np.asarray(y_true).astype(int)
    best, best_t = -1.0, 0.5
    for t in np.round(np.arange(lo, hi + 1e-9, 0.01), 2):
        m = metrics(y_true, np.asarray(proba) >= t)
        p, r = m["precision"], m["recall"]
        fb = (1 + beta**2) * p * r / (beta**2 * p + r) if (beta**2 * p + r) else 0.0
        if fb > best + 1e-12:
            best, best_t = fb, float(t)
    return best_t


def by_group(df: pd.DataFrame, y_col: str, pred_col: str, group_col: str = "platform") -> dict:
    return {g: metrics(d[y_col], d[pred_col]) for g, d in df.groupby(group_col)}
