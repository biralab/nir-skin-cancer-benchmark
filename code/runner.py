"""Cross-validation runners. Every fitted preprocessing step is isolated to the
training fold; grouping is at the patient level (StratifiedGroupKFold).
A spectrum-level mode (plain StratifiedKFold) exists ONLY as a diagnostic to
compare with the v1 manuscript numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import dataio
from preprocess import Chain, MAIN_CHAIN
from models_classical import CLASSICAL, make_classical
from models_deep import DEEP, train_deep

ALL_MODELS = CLASSICAL + DEEP


def make_splitter(mode, seed, n_splits=5):
    if mode == "patient":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    if mode == "spectrum":
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed)
    raise ValueError(mode)


def run_seed_model(seed, model_name, task="binary", split_mode="patient",
                   chain_name=MAIN_CHAIN, log_histories=False):
    """Run one (seed, model) 5-fold CV. Returns (oof_df, hist_df)."""
    if task == "binary":
        X, y, groups, wl, meta = dataio.ufes_binary()
    elif task == "threeclass":
        X, y, groups, wl, meta, order = dataio.ufes_threeclass()
    else:
        raise ValueError(task)
    n_classes = int(y.max()) + 1
    splitter = make_splitter(split_mode, seed)
    split_kwargs = {} if split_mode == "spectrum" else {}
    oof_rows, hist_rows = [], []
    splits = splitter.split(X, y) if split_mode == "spectrum" else splitter.split(X, y, groups)
    for fold, (tr, te) in enumerate(splits):
        chain = Chain(chain_name)
        Xtr = chain.fit_transform(X[tr])
        Xte = chain.transform(X[te])
        if model_name in CLASSICAL:
            clf = make_classical(model_name)
            clf.fit(Xtr, y[tr])
            P = clf.predict_proba(Xte)
            if P.shape[1] == 2 and n_classes == 2:
                P = P[:, 1:]
        else:
            predict_proba, hist = train_deep(
                model_name, Xtr, y[tr],
                groups[tr] if split_mode == "patient" else np.arange(len(tr)),
                seed=seed * 100 + fold)
            P = predict_proba(Xte)
            if n_classes == 2:
                P = P[:, 1:]
            for e in range(len(hist["epoch"])):
                hist_rows.append({
                    "seed": seed, "model": model_name, "fold": fold,
                    "epoch": hist["epoch"][e],
                    "train_loss": hist["train_loss"][e],
                    "val_loss": hist["val_loss"][e],
                    "train_auc": hist["train_auc"][e],
                    "val_auc": hist["val_auc"][e]})
            hist_rows.append({"seed": seed, "model": model_name, "fold": fold,
                              "epoch": -1, "train_loss": hist["best_epoch"],
                              "val_loss": hist["n_epochs_run"],
                              "train_auc": np.nan, "val_auc": np.nan})
        for k, idx in enumerate(te):
            row = {"seed": seed, "model": model_name, "fold": fold,
                   "idx": int(idx), "group": str(groups[idx]),
                   "y": int(y[idx])}
            if n_classes == 2:
                row["p"] = float(P[k, -1] if P.ndim == 2 else P[k])
            else:
                for c in range(n_classes):
                    row[f"p{c}"] = float(P[k, c])
            oof_rows.append(row)
    return pd.DataFrame(oof_rows), pd.DataFrame(hist_rows)


def pooled_metric_by_seed(oof_df, fn):
    """Compute fn(y, p) on pooled OOF per (seed, model)."""
    out = {}
    for (seed, model), g in oof_df.groupby(["seed", "model"]):
        out[(seed, model)] = fn(g["y"].to_numpy(), g["p"].to_numpy())
    return out
