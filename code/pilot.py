"""Pilot: seed 42, binary task, all 7 models, BOTH split modes.
(a) spectrum-level  -> diagnostic: can we reproduce the v1 Table 3 numbers?
(b) patient-level   -> the honest grouped numbers for the revision.
"""
import sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, "/workspace/benchmark")

from runner import run_seed_model, ALL_MODELS
from metrics import metric_set

V1 = {  # v1 Table 3 AUCs for reference
    "Logistic Reg.": 0.835, "PLS-DA+SVM": 0.809, "PCA-LDA": 0.798,
    "Random Forest": 0.761, "1D-CNN": 0.720, "Hybrid CNN-Tr.": 0.670,
    "Pure Transformer": 0.539}

if __name__ == "__main__":
    torch.set_num_threads(3)
    rows = []
    for mode in ["spectrum", "patient"]:
        for model in ALL_MODELS:
            t0 = time.time()
            oof, hist = run_seed_model(42, model, task="binary", split_mode=mode)
            m = metric_set(oof["y"].to_numpy(), oof["p"].to_numpy())
            m.update({"mode": mode, "model": model, "v1_AUC": V1[model],
                      "sec": round(time.time() - t0, 1)})
            if len(hist):
                be = hist[hist["epoch"] == -1]["train_loss"]
                m["median_best_epoch"] = float(be.median()) if len(be) else np.nan
            rows.append(m)
            print(f"[{mode:8s}] {model:18s} AUC={m['AUC']:.3f} (v1 {V1[model]:.3f}) "
                  f"Brier={m['Brier']:.3f} ECE={m['ECE']:.3f} "
                  f"Sens={m['Sens']:.3f} Spec={m['Spec']:.3f} F1={m['F1']:.3f} "
                  f"({m['sec']}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/workspace/benchmark/pilot_results.csv", index=False)
    print("\nSaved pilot_results.csv")
