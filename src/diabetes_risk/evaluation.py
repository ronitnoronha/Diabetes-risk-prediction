"""Metrics, statistical tests, threshold selection and bootstrap confidence intervals."""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score)


def expected_calibration_error(y, p, bins=10):
    """Mean |observed rate − predicted probability|, weighted by bin size (equal-frequency bins)."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return err


def scores(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "ece": expected_calibration_error(y, p)}


def threshold_for_recall(y, p, target):
    """Highest threshold whose recall is still ≥ target (fewest false alarms at that recall)."""
    precision, recall, thr = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= target)[0]
    return float(thr[ok[-1]])


def at_threshold(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"threshold": t, "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
            "specificity": tn / (tn + fp), "accuracy": (tp + tn) / len(y),
            "flagged_per_1000": 1000 * (tp + fp) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for two models scored on the same folds.

    Ordinary t-tests over CV folds are overconfident because training sets overlap; the correction
    inflates the variance by n_test / n_train.
    """
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(d.mean()), float(t), float(p)


def bootstrap_ci(y, p, metric, n=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
