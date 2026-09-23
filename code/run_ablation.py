"""P2e: preprocessing ablation (Reviewer 2, comment 2).
Chains: raw / l2 / l2+snv / l2+snv+SG-deriv. Models: Logistic Reg., 1D-CNN,
Pure Transformer. Patient-level grouped 5-fold CV, seeds 42-44.
"""
import sys, time, os
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/benchmark")
LOCAL = "/workspace/benchmark/out"
MODELS = ["Logistic Reg.", "1D-CNN", "Pure Transformer"]
SEEDS = [42, 43, 44]


def work(args):
    chain_name, model, seed = args
    torch.set_num_threads(1)
    from runner import run_seed_model
    t0 = time.time()
    oof, _ = run_seed_model(seed, model, task="binary", split_mode="patient",
                            chain_name=chain_name)
    auc = roc_auc_score(oof["y"], oof["p"])
    return {"chain": chain_name, "model": model, "seed": seed,
            "AUC": auc, "sec": round(time.time() - t0, 1)}


if __name__ == "__main__":
    os.makedirs(LOCAL, exist_ok=True)
    from preprocess import CHAINS
    tasks = [(c, m, s) for c in CHAINS for m in MODELS for s in SEEDS]
    with Pool(12) as pool:
        rows = []
        for r in pool.imap_unordered(work, tasks):
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{LOCAL}/ablation.csv", index=False)
    print(df.groupby(["chain", "model"])["AUC"].agg(["mean", "std"]))
