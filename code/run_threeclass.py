"""P2d: three-class task (benign / pre-malignant AK / carcinoma BCC+SCC),
patient-level grouped 5-fold CV, seed 42, all 7 models. Saves OOF class
probabilities and deep-model training histories (local disk)."""
import sys, time, os
import pandas as pd
import torch
from multiprocessing import Pool

sys.path.insert(0, "/workspace/benchmark")
LOCAL = "/workspace/benchmark/out"


def tag(m):
    return m.replace(" ", "_").replace("/", "")


def work(model):
    torch.set_num_threads(2)
    from runner import run_seed_model
    t0 = time.time()
    oof, hist = run_seed_model(42, model, task="threeclass", split_mode="patient")
    oof.to_parquet(f"{LOCAL}/oof_3class_{tag(model)}.parquet")
    if len(hist):
        hist.to_parquet(f"{LOCAL}/hist_3class_{tag(model)}.parquet")
    return f"{model} {time.time()-t0:.0f}s"


if __name__ == "__main__":
    os.makedirs(LOCAL, exist_ok=True)
    from runner import ALL_MODELS
    with Pool(7) as pool:
        for msg in pool.imap_unordered(work, ALL_MODELS):
            print(msg, flush=True)
    oofs = [pd.read_parquet(f"{LOCAL}/oof_3class_{tag(m)}.parquet")
            for m in ALL_MODELS]
    pd.concat(oofs).to_parquet(f"{LOCAL}/oof_3class_all.parquet")
    print("3-class DONE")
