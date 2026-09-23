"""P2b: nested CV (outer 5-fold grouped, inner 3-fold grouped grid search) for
the four classical models, binary task, seed 42. Estimates tuning optimism and
reports the chosen hyper-parameter per outer fold."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/benchmark")
import dataio
from preprocess import Chain, MAIN_CHAIN
from models_classical import CLASSICAL, HP_GRIDS, make_classical

if __name__ == "__main__":
    X, y, groups, wl, meta = dataio.ufes_binary()
    rows = []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for model in CLASSICAL:
        grid = [dict(zip(HP_GRIDS[model], v)) for v in
                __import__("itertools").product(*HP_GRIDS[model].values())]
        for ofold, (tr, te) in enumerate(outer.split(X, y, groups)):
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
            best_hp, best_score = None, -np.inf
            for hp in grid:
                scores = []
                for itr, ite in inner.split(X[tr], y[tr], groups[tr]):
                    ch = Chain(MAIN_CHAIN)
                    Xtr = ch.fit_transform(X[tr][itr])
                    Xte = ch.transform(X[tr][ite])
                    clf = make_classical(model, hp)
                    clf.fit(Xtr, y[tr][itr])
                    p = clf.predict_proba(Xte)[:, -1]
                    scores.append(roc_auc_score(y[tr][ite], p))
                if np.mean(scores) > best_score:
                    best_score, best_hp = np.mean(scores), hp
            ch = Chain(MAIN_CHAIN)
            Xtr = ch.fit_transform(X[tr]); Xte = ch.transform(X[te])
            clf = make_classical(model, best_hp)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, -1]
            auc = roc_auc_score(y[te], p)
            rows.append({"model": model, "outer_fold": ofold,
                         "nested_AUC": auc,
                         "chosen_hp": str(list(best_hp.values())[0])})
            print(model, "fold", ofold, "AUC", round(auc, 4),
                  "hp", best_hp, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("/workspace/benchmark/out/nested_cv.csv", index=False)
    print(df.groupby("model")["nested_AUC"].agg(["mean", "std", "min", "max"]))
