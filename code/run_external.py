"""P2c: external validation. Train each model on the FULL UFES binary task
(common wavelength range only) and apply without refitting to Deteccthia
photo-level mean spectra (interpolated onto the UFES grid).
Deep models use a grouped inner split of UFES for early stopping.
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/workspace/benchmark")
import dataio
from preprocess import Chain, MAIN_CHAIN
from models_classical import CLASSICAL, make_classical
from models_deep import DEEP, train_deep

if __name__ == "__main__":
    torch.set_num_threads(8)
    X_u, y_u, g_u, X_d, y_d, wl_common, photos = dataio.external_pair()
    print(f"common channels: {len(wl_common)} ({wl_common.min():.0f}-{wl_common.max():.0f} nm)")
    print(f"UFES train: {X_u.shape}, Deteccthia: {X_d.shape} "
          f"(mal {y_d.sum()} / healthy {(y_d==0).sum()})")
    rows = []
    for model in CLASSICAL + DEEP:
        chain = Chain(MAIN_CHAIN)
        Xu = chain.fit_transform(X_u)
        Xd = chain.transform(X_d)
        if model in CLASSICAL:
            clf = make_classical(model)
            clf.fit(Xu, y_u)
            p = clf.predict_proba(Xd)[:, -1]
            hist = None
        else:
            predict_proba, hist = train_deep(model, Xu, y_u, g_u, seed=42)
            p = predict_proba(Xd)[:, -1]
            if hist:
                h = pd.DataFrame(hist)
                h["model"] = model
                h.to_csv(f"/workspace/benchmark/out/hist_external_{model.replace(' ','_')}.csv", index=False)
        for i in range(len(p)):
            rows.append({"model": model, "photo": photos[i],
                         "y": int(y_d[i]), "p": float(p[i])})
        print(model, "done", flush=True)
    pd.DataFrame(rows).to_csv("/workspace/benchmark/out/external_preds.csv", index=False)
    print("saved external_preds.csv")
