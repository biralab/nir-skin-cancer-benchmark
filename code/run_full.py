"""P2a: full binary benchmark — 20 seeds (42-61) x 7 models, patient-level
grouped 5-fold CV. Parallel over (seed, model). Workers write parquet to LOCAL
/workspace (S3 mounts do not support parquet writes); the main process combines
and then copies the two combined files to shared-workspace."""
import os, sys, time, shutil
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, "/workspace/benchmark")
LOCAL = "/workspace/benchmark/out"
OUT = "/mnt/shared-workspace/shared/jdim_r1"
SEEDS = list(range(42, 62))


def tag(m):
    return m.replace(" ", "_").replace("/", "")


def work(args):
    seed, model = args
    import torch
    torch.set_num_threads(1)
    from runner import run_seed_model
    t0 = time.time()
    oof, hist = run_seed_model(seed, model, task="binary", split_mode="patient")
    oof.to_parquet(f"{LOCAL}/oof_binary_{seed}_{tag(model)}.parquet")
    if len(hist):
        hist.to_parquet(f"{LOCAL}/hist_binary_{seed}_{tag(model)}.parquet")
    return f"{seed} {model} done in {time.time()-t0:.0f}s"


if __name__ == "__main__":
    os.makedirs(LOCAL, exist_ok=True)
    from runner import ALL_MODELS
    tasks = [(s, m) for s in SEEDS for m in ALL_MODELS]
    with Pool(12) as pool:
        for msg in pool.imap_unordered(work, tasks):
            print(msg, flush=True)
    oofs, hists = [], []
    for s in SEEDS:
        for m in ALL_MODELS:
            f = f"{LOCAL}/oof_binary_{s}_{tag(m)}.parquet"
            if os.path.exists(f):
                oofs.append(pd.read_parquet(f))
            h = f"{LOCAL}/hist_binary_{s}_{tag(m)}.parquet"
            if os.path.exists(h):
                hists.append(pd.read_parquet(h))
    big = pd.concat(oofs)
    big.to_parquet(f"{LOCAL}/oof_binary_all.parquet")
    shutil.copy(f"{LOCAL}/oof_binary_all.parquet", f"{OUT}/oof_binary_all.parquet")
    if hists:
        hh = pd.concat(hists)
        hh.to_parquet(f"{LOCAL}/hist_binary_all.parquet")
        shutil.copy(f"{LOCAL}/hist_binary_all.parquet", f"{OUT}/hist_binary_all.parquet")
    print("ALL DONE", len(big), "oof rows,", sum(len(h) for h in hists), "history rows")
