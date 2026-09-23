"""Recompute all DeLong-dependent outputs with the fixed implementation:
table3_delong.csv, table4_delong.csv, fig5b_tests.csv, table7_tests.csv (AUC row).
Bootstrap-based outputs are unchanged."""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/benchmark")
from metrics import delong_test, paired_bootstrap_diff, holm_adjust

OUT = "/workspace/benchmark/tables"
MODEL_ORDER = ["Logistic Reg.", "PLS-DA+SVM", "PCA-LDA", "Random Forest",
               "1D-CNN", "Hybrid CNN-Tr.", "Pure Transformer"]
CLASSES = ["benign", "pre-malignant (AK)", "carcinoma (BCC/SCC)"]
REF = "Logistic Reg."

oof = pd.read_parquet("/mnt/shared-workspace/shared/jdim_r1/oof_binary_all.parquet")
d = oof[oof.seed == 42]

# --- Table 3 DeLong matrix ---
M = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
for a in MODEL_ORDER:
    for b in MODEL_ORDER:
        ga, gb = d[d.model == a], d[d.model == b]
        M.loc[a, b] = delong_test(ga.y.to_numpy(), ga.p.to_numpy(), gb.p.to_numpy())[2]
M.to_csv(f"{OUT}/table3_delong.csv")
print("Table 3 DeLong matrix (p-values):"); print(M.round(4).to_string())

# --- Table 4 DeLong matrix (external) ---
e = pd.read_csv("/workspace/benchmark/out/external_preds.csv")
M4 = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
for a in MODEL_ORDER:
    for b in MODEL_ORDER:
        ga, gb = e[e.model == a], e[e.model == b]
        M4.loc[a, b] = delong_test(ga.y.to_numpy(), ga.p.to_numpy(), gb.p.to_numpy())[2]
M4.to_csv(f"{OUT}/table4_delong.csv")
print("\nTable 4 DeLong matrix (external):"); print(M4.round(4).to_string())

# --- Fig 5b tests (three-class, vs best classical per class) ---
o3 = pd.read_parquet("/workspace/benchmark/out/oof_3class_all.parquet")
trows = []
for c in range(3):
    raw = {}
    gb = o3[o3.model == REF]
    yb = (gb.y.to_numpy() == c).astype(int); pb = gb[f"p{c}"].to_numpy()
    grb = gb.group.to_numpy()
    for model in MODEL_ORDER:
        if model == REF:
            continue
        g = o3[o3.model == model]
        p_dl = delong_test(yb, pb, g[f"p{c}"].to_numpy())[2]
        _, _, _, p_bs = paired_bootstrap_diff(
            yb, pb, g[f"p{c}"].to_numpy(), grb,
            lambda yy, pp: roc_auc_score(yy, pp), n_boot=2000, seed=1)
        raw[model] = (p_dl, p_bs)
    adj_dl = holm_adjust({m: v[0] for m, v in raw.items()})
    adj_bs = holm_adjust({m: v[1] for m, v in raw.items()})
    for model in raw:
        trows.append({"Class": CLASSES[c], "vs": model,
                      "p_delong": round(raw[model][0], 5),
                      "p_delong_holm": round(adj_dl[model], 5),
                      "p_boot": round(raw[model][1], 5),
                      "p_boot_holm": round(adj_bs[model], 5)})
tt = pd.DataFrame(trows)
tt.to_csv(f"{OUT}/fig5b_tests.csv", index=False)
print("\nFig 5b tests (vs Logistic Reg.):"); print(tt.to_string(index=False))

# --- Table 7 AUC row (raw vs cross-fitted Platt) ---
from sklearn.linear_model import LogisticRegression as PlattLR
from sklearn.model_selection import StratifiedGroupKFold
dd = d[d.model == REF].reset_index(drop=True)
y = dd.y.to_numpy(); p = dd.p.to_numpy(); g = dd.group.to_numpy()
eps = 1e-6
logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
p_cal = np.zeros_like(p)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
for tr, te in skf.split(logit.reshape(-1, 1), y, g):
    pl = PlattLR(C=1e10, solver="lbfgs").fit(logit[tr].reshape(-1, 1), y[tr])
    p_cal[te] = pl.predict_proba(logit[te].reshape(-1, 1))[:, 1]
a1, a2, p_dl7 = delong_test(y, p, p_cal)
print(f"\nTable 7 AUC raw {a1:.4f} vs Platt {a2:.4f}, DeLong p = {p_dl7:.4f}")
t7t = pd.read_csv(f"{OUT}/table7_tests.csv")
t7t.loc[t7t.metric == "AUC", "p"] = round(p_dl7, 4)
t7t.to_csv(f"{OUT}/table7_tests.csv", index=False)
np.save("/workspace/benchmark/out/p_cal_seed42.npy", p_cal)
print("done")
