"""Supplementary Table S2: external-validation sensitivity to cohort-level
re-standardisation. Models are trained on full UFES exactly as in
run_external.py; the Deteccthia spectra are then preprocessed with the chain
statistics fitted on Deteccthia itself (per-cohort re-standardisation)
instead of importing the UFES-fitted scaler.
"""
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, "/workspace/benchmark")
import dataio
from preprocess import Chain, MAIN_CHAIN
from models_classical import CLASSICAL, make_classical
from models_deep import DEEP, train_deep

if __name__ == "__main__":
    torch.set_num_threads(8)
    X_u, y_u, g_u, X_d, y_d, wl_common, photos = dataio.external_pair()
    rows = []
    for model in CLASSICAL + DEEP:
        chain_u = Chain(MAIN_CHAIN)
        Xu = chain_u.fit_transform(X_u)
        # cohort-level re-standardisation of the external cohort
        chain_d = Chain(MAIN_CHAIN)
        Xd_re = chain_d.fit_transform(X_d)
        if model in CLASSICAL:
            clf = make_classical(model)
            clf.fit(Xu, y_u)
            p = clf.predict_proba(Xd_re)[:, -1]
        else:
            predict_proba, _ = train_deep(model, Xu, y_u, g_u, seed=42)
            p = predict_proba(Xd_re)[:, -1]
        auc = roc_auc_score(y_d, p)
        brier = brier_score_loss(y_d, np.clip(p, 0, 1))
        rows.append({"model": model, "AUC_restd": auc, "Brier_restd": brier})
        print(model, f"AUC={auc:.3f} Brier={brier:.3f}", flush=True)
        pd.DataFrame({"model": model, "photo": photos, "y": y_d, "p": p}).to_csv(
            f"/workspace/benchmark/out/external_restd_preds_{model.replace(' ','_')}.csv",
            index=False)
    pd.DataFrame(rows).to_csv("/workspace/benchmark/out/external_restd.csv", index=False)
    print("saved external_restd.csv")
