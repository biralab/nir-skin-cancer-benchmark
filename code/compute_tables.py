"""P3: compute all revised tables' statistics from saved OOF predictions.

Outputs (CSV, /workspace/benchmark/tables/):
  table3_main.csv        metrics per model (seed 42 pooled OOF) + group-bootstrap 95% CIs
  table3_tests.csv       paired group-bootstrap vs Logistic Reg. per metric (Holm within family)
  table3_delong.csv      pairwise DeLong matrix (seed 42)
  table4_external.csv    external-cohort metrics + CIs; table4_delong.csv pairwise DeLong
  table5_seeds.csv       20-seed mean±SD [min,max]; table5_tests.csv Friedman/Wilcoxon-Holm/Levene
  table6_nested.csv      nested CV per outer fold + 95% CI + one-sample test vs single-split
  table7_platt.csv       raw vs cross-fitted Platt: metrics + paired bootstrap + DeLong
  dca_curves.csv         decision-curve data (net benefit vs threshold) for calibrated LogReg + 1D-CNN
  dca_clinical.csv       quantified clinical utility at selected thresholds (R2.4)
All resampling is at the patient/photo (group) level. alpha=0.05, Holm within
each metric family. Every number derives from the stored OOF predictions.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.linear_model import LogisticRegression as PlattLR
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats

sys.path.insert(0, "/workspace/benchmark")
from metrics import metric_set, ece_score, bootstrap_ci, paired_bootstrap_diff, \
    delong_test, friedman_then_wilcoxon, holm_adjust

OUT = "/workspace/benchmark/tables"
os.makedirs(OUT, exist_ok=True)
SH = "/mnt/shared-workspace/shared/jdim_r1"
REF = "Logistic Reg."
MODEL_ORDER = ["Logistic Reg.", "PLS-DA+SVM", "PCA-LDA", "Random Forest",
               "1D-CNN", "Hybrid CNN-Tr.", "Pure Transformer"]
METRICS = ["AUC", "PR-AUC", "Brier", "ECE", "Sens", "Spec", "F1"]
METRIC_FNS = {
    "AUC": lambda y, p: roc_auc_score(y, p),
    "PR-AUC": lambda y, p: average_precision_score(y, p),
    "Brier": lambda y, p: brier_score_loss(y, p),
    "ECE": lambda y, p: ece_score(y, p, 15),
    "Sens": lambda y, p: ((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1),
    "Spec": lambda y, p: ((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1),
    "F1": lambda y, p: f1_score(y, (p >= 0.5).astype(int), zero_division=0),
}


def table3(oof):
    print("== Table 3 (seed 42, pooled OOF, patient-level bootstrap CIs) ==")
    d = oof[oof.seed == 42]
    rows, tests = [], {m: {} for m in METRICS}
    ref = d[d.model == REF]
    y_r, p_r = ref.y.to_numpy(), ref.p.to_numpy()
    for model in MODEL_ORDER:
        g = d[d.model == model]
        y, p, gr = g.y.to_numpy(), g.p.to_numpy(), g.group.to_numpy()
        ms = metric_set(y, p)
        row = {"Model": model, "Family": "C" if model in MODEL_ORDER[:4] else "D"}
        for mname in METRICS:
            lo, hi = bootstrap_ci(y, p, gr, METRIC_FNS[mname], n_boot=2000, seed=1)
            row[mname] = round(ms[mname], 3)
            row[mname + "_CI"] = f"[{lo:.3f},{hi:.3f}]"
            if model != REF:
                diff, dlo, dhi, pp = paired_bootstrap_diff(
                    y, p_r, p, gr, METRIC_FNS[mname], n_boot=2000, seed=1)
                tests[mname][model] = pp
        rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(f"{OUT}/table3_main.csv", index=False)
    trows = []
    for mname in METRICS:
        adj = holm_adjust(tests[mname])
        for model, p in adj.items():
            trows.append({"metric": mname, "vs": model,
                          "p_raw": tests[mname][model], "p_holm": adj[model]})
    pd.DataFrame(trows).to_csv(f"{OUT}/table3_tests.csv", index=False)
    M = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for a in MODEL_ORDER:
        for b in MODEL_ORDER:
            ga = d[d.model == a]; gb = d[d.model == b]
            _, _, p = delong_test(ga.y.to_numpy(), ga.p.to_numpy(), gb.p.to_numpy())
            M.loc[a, b] = p
    M.to_csv(f"{OUT}/table3_delong.csv")
    print(t3[["Model", "AUC", "AUC_CI", "Brier", "ECE", "Sens", "Spec", "F1"]].to_string(index=False))
    return t3


def table4():
    print("== Table 4 (external Deteccthia, n=152 photo-level) ==")
    d = pd.read_csv("/workspace/benchmark/out/external_preds.csv")
    rows = []
    M = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for model in MODEL_ORDER:
        g = d[d.model == model]
        y, p, gr = g.y.to_numpy(), g.p.to_numpy(), g.photo.to_numpy()
        auc = roc_auc_score(y, p); br = brier_score_loss(y, p)
        alo, ahi = bootstrap_ci(y, p, gr, METRIC_FNS["AUC"], n_boot=2000, seed=1)
        blo, bhi = bootstrap_ci(y, p, gr, METRIC_FNS["Brier"], n_boot=2000, seed=1)
        rows.append({"Model": model, "AUC": round(auc, 3),
                     "AUC_CI": f"[{alo:.3f},{ahi:.3f}]", "Brier": round(br, 3),
                     "Brier_CI": f"[{blo:.3f},{bhi:.3f}]"})
    for a in MODEL_ORDER:
        for b in MODEL_ORDER:
            ga = d[d.model == a]; gb = d[d.model == b]
            _, _, p = delong_test(ga.y.to_numpy(), ga.p.to_numpy(), gb.p.to_numpy())
            M.loc[a, b] = p
    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{OUT}/table4_external.csv", index=False)
    M.to_csv(f"{OUT}/table4_delong.csv")
    print(t4.to_string(index=False))
    return t4


def table5(oof):
    print("== Table 5 (20 seeds) ==")
    aucs = {}
    for (seed, model), g in oof.groupby(["seed", "model"]):
        aucs.setdefault(model, []).append((seed, roc_auc_score(g.y, g.p)))
    rows = []
    mat = np.zeros((20, len(MODEL_ORDER)))
    for j, model in enumerate(MODEL_ORDER):
        vals = np.array([v for s, v in sorted(aucs[model])])
        mat[:, j] = vals
        rows.append({"Model": model, "Mean": round(vals.mean(), 4),
                     "SD": round(vals.std(ddof=1), 4),
                     "min": round(vals.min(), 3), "max": round(vals.max(), 3)})
    t5 = pd.DataFrame(rows)
    t5.to_csv(f"{OUT}/table5_seeds.csv", index=False)
    wins = int(sum(1 for i in range(20) if mat[i, :4].max() > mat[i, 4:].max()))
    fried_p, wilcox = friedman_then_wilcoxon(mat, MODEL_ORDER, REF)
    lev_stat, lev_p = stats.levene(mat[:, :4].flatten(), mat[:, 4:].flatten())
    pd.DataFrame([{"test": "Friedman (7 models x 20 seeds)", "p": fried_p},
                  {"test": "Levene (classical vs deep variance)", "p": lev_p}] +
                 [{"test": f"Wilcoxon vs {REF}: {k}", "p": v}
                  for k, v in wilcox.items()]).to_csv(f"{OUT}/table5_tests.csv", index=False)
    print(t5.to_string(index=False))
    print(f"best-classical > best-deep in {wins}/20 seeds; Friedman p={fried_p:.2e}; Levene p={lev_p:.2e}")
    return t5, wins, fried_p, lev_p, wilcox


def table6(oof):
    print("== Table 6 (nested CV vs single-split) ==")
    nc = pd.read_csv("/workspace/benchmark/out/nested_cv.csv")
    d = oof[oof.seed == 42]
    rows = []
    for model in ["Logistic Reg.", "PLS-DA+SVM", "PCA-LDA", "Random Forest"]:
        g = d[d.model == model]
        single = roc_auc_score(g.y, g.p)
        vals = nc[nc.model == model]["nested_AUC"].to_numpy()
        mean, sd = vals.mean(), vals.std(ddof=1)
        ci = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=sd / np.sqrt(len(vals)))
        try:
            p = stats.wilcoxon(vals - single).pvalue
        except ValueError:
            p = 1.0
        rows.append({"Model": model, "Nested_mean": round(mean, 3),
                     "Nested_SD": round(sd, 3),
                     "Nested_CI": f"[{ci[0]:.3f},{ci[1]:.3f}]",
                     "Single_split": round(single, 3),
                     "abs_delta": round(abs(mean - single), 3),
                     "p_vs_single": round(p, 3),
                     "chosen_hp": "; ".join(str(v) for v in nc[nc.model == model]["chosen_hp"])})
    t6 = pd.DataFrame(rows)
    t6.to_csv(f"{OUT}/table6_nested.csv", index=False)
    print(t6.to_string(index=False))
    return t6


def table7(oof):
    print("== Table 7 (Platt scaling, cross-fitted on seed-42 OOF) ==")
    d = oof[(oof.seed == 42) & (oof.model == REF)].reset_index(drop=True)
    y = d.y.to_numpy(); p = d.p.to_numpy(); g = d.group.to_numpy()
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    p_cal = np.zeros_like(p)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    for tr, te in skf.split(logit.reshape(-1, 1), y, g):
        pl = PlattLR(C=1e10, solver="lbfgs")
        pl.fit(logit[tr].reshape(-1, 1), y[tr])
        p_cal[te] = pl.predict_proba(logit[te].reshape(-1, 1))[:, 1]
    rows = []
    for label, pp in [("Raw probabilities", p), ("After Platt scaling", p_cal)]:
        ms = metric_set(y, pp)
        rows.append({"Setting": label, "AUC": round(ms["AUC"], 4),
                     "Brier": round(ms["Brier"], 4), "ECE": round(ms["ECE"], 4)})
    tb = []
    for mname in ["Brier", "ECE"]:
        diff, lo, hi, pp_ = paired_bootstrap_diff(y, p, p_cal, g, METRIC_FNS[mname],
                                                  n_boot=5000, seed=1)
        tb.append({"metric": mname, "diff": round(diff, 4),
                   "CI": f"[{lo:.4f},{hi:.4f}]", "p": pp_})
    _, _, pdl = delong_test(y, p, p_cal)
    tb.append({"metric": "AUC", "diff": 0.0, "CI": "-", "p": pdl})
    t7 = pd.DataFrame(rows)
    t7.to_csv(f"{OUT}/table7_platt.csv", index=False)
    pd.DataFrame(tb).to_csv(f"{OUT}/table7_tests.csv", index=False)
    print(t7.to_string(index=False)); print(pd.DataFrame(tb).to_string(index=False))
    return t7, p_cal


def dca(oof, p_cal):
    print("== Decision-curve data + clinical quantification (R2.4) ==")
    d42 = oof[oof.seed == 42]
    y = d42[d42.model == REF].y.to_numpy()
    n = len(y); prev = y.mean()
    p_cnn = d42[d42.model == "1D-CNN"].p.to_numpy()
    thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2)
    rows = []
    for pt in thresholds:
        for name, pp in [("LogReg calibrated", p_cal), ("1D-CNN", p_cnn)]:
            pred = pp >= pt
            tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
            nb = tp / n - fp / n * pt / (1 - pt)
            rows.append({"threshold": pt, "model": name, "net_benefit": nb})
        nb_all = prev - (1 - prev) * pt / (1 - pt)
        rows.append({"threshold": pt, "model": "Treat all", "net_benefit": nb_all})
        rows.append({"threshold": pt, "model": "Treat none", "net_benefit": 0.0})
    pd.DataFrame(rows).to_csv(f"{OUT}/dca_curves.csv", index=False)
    clin = []
    for pt in [0.10, 0.20, 0.30]:
        pred = p_cal >= pt
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1)
        clin.append({"threshold": pt, "sensitivity": round(sens, 3),
                     "biopsies_avoided_per_1000": round(tn / n * 1000, 0),
                     "cancers_missed_per_1000": round(fn / n * 1000, 1),
                     "NNT_biopsy_per_cancer_detected": round((tp + fp) / max(tp, 1), 2)})
    c = pd.DataFrame(clin)
    c.to_csv(f"{OUT}/dca_clinical.csv", index=False)
    print(c.to_string(index=False))
    return c


if __name__ == "__main__":
    oof = pd.read_parquet(f"{SH}/oof_binary_all.parquet")
    t3 = table3(oof)
    t4 = table4()
    t5, wins, fried_p, lev_p, wilcox = table5(oof)
    t6 = table6(oof)
    t7, p_cal = table7(oof)
    c = dca(oof, p_cal)
    print("\nAll tables written to", OUT)
