"""Evaluation metrics + statistical tests for the benchmark.

All metrics are computed on pooled out-of-fold (OOF) predictions.
Confidence intervals and paired tests resample at the GROUP (patient/photo)
level to respect the clustered structure of replicate spectra.
DeLong tests (spectrum-level, conventional) are also provided for AUC.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, recall_score)
from scipy import stats


def ece_score(y, p, n_bins=15):
    """Expected calibration error, equal-width bins on [0,1]."""
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metric_set(y, p, n_bins=15):
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "PR-AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece_score(y, p, n_bins),
        "Sens": float(recall_score(y, pred, zero_division=0)),
        "Spec": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


# ---------------- bootstrap (group-aware) ----------------

def _group_resample_idx(groups, rng):
    ug = np.unique(groups)
    drawn = rng.choice(ug, size=len(ug), replace=True)
    return np.concatenate([np.flatnonzero(groups == g) for g in drawn])


def bootstrap_ci(y, p, groups, fn, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of metric fn(y,p), resampling groups."""
    rng = np.random.default_rng(seed)
    vals = []
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    for _ in range(n_boot):
        ii = _group_resample_idx(groups, rng)
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(fn(y[ii], p[ii]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(y, p_ref, p_alt, groups, fn, n_boot=10000, seed=0):
    """Paired group-bootstrap of fn(alt)-fn(ref): returns (diff, lo, hi, p_two-sided).

    p is the two-sided percentile bootstrap p-value for H0: diff = 0.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p_ref = np.asarray(p_ref); p_alt = np.asarray(p_alt)
    groups = np.asarray(groups)
    diffs = []
    for _ in range(n_boot):
        ii = _group_resample_idx(groups, rng)
        if len(np.unique(y[ii])) < 2:
            continue
        diffs.append(fn(y[ii], p_alt[ii]) - fn(y[ii], p_ref[ii]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(np.mean(diffs)), float(lo), float(hi), float(min(p, 1.0))


# ---------------- DeLong ----------------

def _midranks(x):
    """Midranks (1-based) returned in the ORIGINAL element order; ties averaged."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks_sorted = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks_sorted
    return out


def _structural_components(pos_scores, neg_scores):
    """DeLong structural components.
    V10_i = (R_i^{all} - R_i^{pos}) / n_neg  for each positive i
    V01_j = (R_j^{all} - R_j^{neg}) / m_pos  for each negative j
    AUC = mean(V10) = 1 - mean(V01) (with midranks handling ties).
    """
    m, n = len(pos_scores), len(neg_scores)
    joined = np.concatenate([pos_scores, neg_scores])
    r_all = _midranks(joined)
    r_pos = _midranks(pos_scores)
    r_neg = _midranks(neg_scores)
    v10 = (r_all[:m] - r_pos) / n
    v01 = (r_all[m:] - r_neg) / m
    return v10, v01


def delong_test(y, p1, p2):
    """Two-sided DeLong test for correlated AUCs. Returns (auc1, auc2, p)."""
    y = np.asarray(y)
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    v10_1, v01_1 = _structural_components(p1[pos], p1[~pos])
    v10_2, v01_2 = _structural_components(p2[pos], p2[~pos])
    auc1 = v10_1.mean(); auc2 = v10_2.mean()
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 0:
        return float(auc1), float(auc2), 1.0
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(auc1), float(auc2), float(p)


# ---------------- multi-seed tests ----------------

def friedman_then_wilcoxon(seed_matrix, model_names, ref="Logistic Reg."):
    """seed_matrix: (n_seeds, n_models) AUCs. Returns Friedman p and
    Holm-adjusted Wilcoxon signed-rank p of each model vs ref."""
    from scipy.stats import friedmanchisquare, wilcoxon
    F = friedmanchisquare(*[seed_matrix[:, j] for j in range(seed_matrix.shape[1])])
    out = {}
    jref = model_names.index(ref)
    raw = {}
    for j, name in enumerate(model_names):
        if name == ref:
            continue
        d = seed_matrix[:, jref] - seed_matrix[:, j]
        try:
            w = wilcoxon(d, alternative="greater", zero_method="wilcox")
            raw[name] = w.pvalue
        except ValueError:
            raw[name] = 1.0
    # Holm
    order = sorted(raw, key=raw.get)
    adj = {}
    m = len(order)
    running = 0.0
    for rank, name in enumerate(order):
        val = min((m - rank) * raw[name], 1.0)
        running = max(running, val)
        adj[name] = running
    return float(F.pvalue), adj


def holm_adjust(pvals: dict):
    order = sorted(pvals, key=pvals.get)
    adj = {}; m = len(order); running = 0.0
    for rank, name in enumerate(order):
        val = min((m - rank) * pvals[name], 1.0)
        running = max(running, val)
        adj[name] = running
    return adj
