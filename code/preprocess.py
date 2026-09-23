"""Fold-isolated preprocessing chains for vibrational spectra.

Main chain (as in the manuscript): L2 row-normalisation -> SNV (per-spectrum)
-> per-channel standardisation fitted on the TRAINING fold only.
L2 and SNV are row-wise and leak nothing; the scaler is the only fitted step.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def l2_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def snv(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (X - m) / s


def sg_deriv(X, window=11, poly=3, deriv=1):
    from scipy.signal import savgol_filter
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=deriv, axis=1)


class Chain:
    """A named preprocessing chain; fit on train, transform any set."""

    def __init__(self, name: str):
        self.name = name
        self.scaler = None

    def _row_steps(self, X):
        if self.name == "raw":
            return X
        if self.name == "l2":
            return l2_rows(X)
        if self.name == "l2_snv":
            return snv(l2_rows(X))
        if self.name == "l2_snv_sg":
            return sg_deriv(snv(l2_rows(X)))
        raise ValueError(self.name)

    def fit_transform(self, Xtr):
        Z = self._row_steps(Xtr)
        self.scaler = StandardScaler().fit(Z)
        return self.scaler.transform(Z)

    def transform(self, X):
        return self.scaler.transform(self._row_steps(X))


CHAINS = ["raw", "l2", "l2_snv", "l2_snv_sg"]
MAIN_CHAIN = "l2_snv"
