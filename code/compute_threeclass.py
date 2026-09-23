"""P3b: three-class statistics (revised Fig 5 / old Fig 9b) + DeLong sanity check.

Per-class one-vs-rest AUC for all 7 models with patient-level bootstrap 95% CIs;
DeLong + paired bootstrap tests vs the best classical model per class (Holm within
class); row-normalised confusion matrix for logistic regression; macro-AUC per model.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, "/workspace/benchmark")
from metrics import bootstrap_ci, paired_bootstrap_diff, delong_test, holm_adjust

OUT = "/workspace/benchmark/tables"
LOCAL = "/workspace/benchmark/out"
MODEL_ORDER = ["Logistic Reg.", "PLS-DA+SVM", "PCA-LDA", "Random Forest",
               "1D-CNN", "Hybrid CNN-Tr.", "Pure Transformer"]
CLASSES = ["benign", "pre-malignant (AK)", "carcinoma (BCC/SCC)"]

oof = pd.read_parquet(f"{LOCAL}/oof_3class_all.parquet")
print("rows:", len(oof), "models:", oof.model.nunique())

rows = []
per_model_class_auc = {}
for model in MODEL_ORDER:
    g = oof[oof.model == model]
    y = g.y.to_numpy(); gr = g.group.to_numpy()
    P = g[["p0", "p1", "p2"]].to_numpy()
    aucs = []
    for c in range(3):
        yb = (y == c).astype(int)
        auc = roc_auc_score(yb, P[:, c])
        lo, hi = bootstrap_ci(yb, P[:, c], gr,
                              lambda yy, pp: roc_auc_score(yy, pp),
                              n_boot=2000, seed=1)
        aucs.append(auc)
        rows.append({"Model": model, "Class": CLASSES[c], "AUC": round(auc, 3),
                     "CI": f"[{lo:.3f},{hi:.3f}]"})
    per_model_class_auc[model] = aucs
    print(model, [round(a, 3) for a in aucs], "macro", round(np.mean(aucs), 3))

t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/fig5b_perclass_auc.csv", index=False)

# tests vs best classical per class (best classical = max macro among first 4)
macro = {m: np.mean(v) for m, v in per_model_class_auc.items()}
best_classical = max(MODEL_ORDER[:4], key=lambda m: macro[m])
print("best classical (macro):", best_classical)
trows = []
for c in range(3):
    raw = {}
    gb = oof[oof.model == best_classical]
    yb = (gb.y.to_numpy() == c).astype(int); pb = gb[f"p{c}"].to_numpy()
    grb = gb.group.to_numpy()
    for model in MODEL_ORDER:
        if model == best_classical:
            continue
        g = oof[oof.model == model]
        _, _, p_dl = delong_test(yb, pb, g[f"p{c}"].to_numpy())
        _, _, _, p_bs = paired_bootstrap_diff(
            yb, pb, g[f"p{c}"].to_numpy(), grb,
            lambda yy, pp: roc_auc_score(yy, pp), n_boot=2000, seed=1)
        raw[model] = (p_dl, p_bs)
    adj_dl = holm_adjust({m: v[0] for m, v in raw.items()})
    adj_bs = holm_adjust({m: v[1] for m, v in raw.items()})
    for model in raw:
        trows.append({"Class": CLASSES[c], "vs": model,
                      "p_delong": raw[model][0], "p_delong_holm": adj_dl[model],
                      "p_boot": raw[model][1], "p_boot_holm": adj_bs[model]})
tt = pd.DataFrame(trows)
tt.to_csv(f"{OUT}/fig5b_tests.csv", index=False)
print(tt.to_string(index=False))

# confusion matrix for logistic regression
g = oof[oof.model == "Logistic Reg."]
y = g.y.to_numpy(); pred = g[["p0", "p1", "p2"]].to_numpy().argmax(1)
cm = confusion_matrix(y, pred, labels=[0, 1, 2])
cm_norm = cm / cm.sum(1, keepdims=True)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(f"{OUT}/fig5a_confusion.csv")
print("confusion (counts):"); print(cm)
print("confusion (row-normalised):"); print(np.round(cm_norm, 3))

# DeLong sanity check on binary seed 42: LR vs 1D-CNN, DeLong vs bootstrap
sh = pd.read_parquet("/mnt/shared-workspace/shared/jdim_r1/oof_binary_all.parquet")
d = sh[sh.seed == 42]
a = d[d.model == "Logistic Reg."]; b = d[d.model == "1D-CNN"]
_, _, p_dl = delong_test(a.y.to_numpy(), a.p.to_numpy(), b.p.to_numpy())
_, _, _, p_bs = paired_bootstrap_diff(a.y.to_numpy(), a.p.to_numpy(), b.p.to_numpy(),
                                      a.group.to_numpy(),
                                      lambda yy, pp: roc_auc_score(yy, pp),
                                      n_boot=5000, seed=1)
print(f"\nDeLong sanity (LR vs 1D-CNN, seed42): DeLong p={p_dl:.2e}, bootstrap p={p_bs:.2e}")
