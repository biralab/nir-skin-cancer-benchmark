"""P4: revised figures. Main: Fig3 DeLong matrix, Fig4 DCA, Fig5 three-class,
Fig6 learning process. Supplementary: S1 ROC, S2 external, S3 seed stability,
S4 calibration, S5 ablation. Liberation Sans, Okabe-Ito, PNG 300dpi + SVG.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, "/workspace/benchmark")
TAB = "/workspace/benchmark/tables"
FIG = "/workspace/benchmark/figures"
os.makedirs(FIG, exist_ok=True)

MODEL_ORDER = ["Logistic Reg.", "PLS-DA+SVM", "PCA-LDA", "Random Forest",
               "1D-CNN", "Hybrid CNN-Tr.", "Pure Transformer"]
C_CLASSICAL = "#0072B2"   # Okabe-Ito blue
C_DEEP = "#E69F00"        # Okabe-Ito orange
C_GREY = "#999999"
FAM = ["C", "C", "C", "C", "D", "D", "D"]


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------- Fig 3: pairwise DeLong matrix ----------
def fig3():
    M = pd.read_csv(f"{TAB}/table3_delong.csv", index_col=0).loc[MODEL_ORDER, MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    logp = -np.log10(M.astype(float).clip(lower=1e-16))
    im = ax.imshow(logp, cmap="viridis", vmin=0, vmax=8)
    ax.set_xticks(range(7), MODEL_ORDER, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(7), MODEL_ORDER, fontsize=9)
    for i in range(7):
        for j in range(7):
            p = M.iloc[i, j]
            if i == j:
                txt = "—"
            else:
                txt = f"{p:.3f}" if p >= 0.001 else "<0.001"
                if p < 0.05:
                    txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if logp.iloc[i, j] > 4 else "black")
    ax.set_title("Pairwise DeLong tests of correlated AUCs (NIR-SC-UFES, binary)",
                 fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(r"$-\log_{10}(p)$", fontsize=9)
    ax.text(0, -0.85, "", fontsize=8)  # spacing
    save(fig, "Fig3_delong_matrix")


# ---------- Fig 4: decision-curve analysis ----------
def fig4():
    d = pd.read_csv(f"{TAB}/dca_curves.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    styles = {"LogReg calibrated": (C_CLASSICAL, "-", "Logistic regression (calibrated)"),
              "1D-CNN": (C_DEEP, "-", "1D-CNN"),
              "Treat all": (C_GREY, "--", "Treat all"),
              "Treat none": ("black", ":", "Treat none")}
    for name, (c, ls, lab) in styles.items():
        g = d[d.model == name]
        ax.plot(g.threshold, g.net_benefit, ls, color=c, lw=2, label=lab)
    ax.set_xlabel("Threshold probability", fontsize=10)
    ax.set_ylabel("Net benefit", fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, max(d.net_benefit.max() * 1.05, 0.05))
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Decision-curve analysis (NIR-SC-UFES, patient-level OOF)",
                 fontsize=11)
    save(fig, "Fig4_dca")


# ---------- Fig 5: three-class ----------
def fig5():
    cm = pd.read_csv(f"{TAB}/fig5a_confusion.csv", index_col=0)
    auc = pd.read_csv(f"{TAB}/fig5b_perclass_auc.csv")
    tests = pd.read_csv(f"{TAB}/fig5b_tests.csv")
    classes = list(cm.index)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) confusion matrix
    ax = axes[0]
    cmn = cm.to_numpy() / cm.to_numpy().sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), classes, fontsize=9)
    ax.set_yticks(range(3), classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("True", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm.to_numpy()[i,j]}\n({cmn[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_title("(a) Three-class confusion matrix — Logistic Reg.\n"
                 "row-normalised (n=641 spectra / 313 patients)", fontsize=10)
    # (b) per-class OvR AUC with CI + significance
    ax = axes[1]
    x = np.arange(3)
    width = 0.11
    macros = auc.groupby("Model")["AUC"].mean()
    for k, model in enumerate(MODEL_ORDER):
        g = auc[auc.Model == model]
        vals = g.set_index("Class").loc[classes, "AUC"].to_numpy()
        cis = g.set_index("Class").loc[classes, "CI"].str.strip("[]").str.split(",", expand=True).astype(float)
        err = np.vstack([vals - cis[0].to_numpy(), cis[1].to_numpy() - vals])
        col = C_CLASSICAL if model in MODEL_ORDER[:4] else C_DEEP
        alpha = 1.0 - 0.13 * (k % 4)
        ax.bar(x + (k - 3) * width, vals, width * 0.92, color=col, alpha=alpha,
               label=f"{model} (macro {macros[model]:.3f})",
               yerr=err, capsize=1.5, error_kw=dict(lw=0.7, ecolor="black"))
    # significance stars vs Logistic Reg. (Holm bootstrap)
    for ci_, cls in enumerate(classes):
        sub = tests[tests.Class == cls]
        for k, model in enumerate(MODEL_ORDER[1:], start=1):
            row = sub[sub.vs == model]
            if len(row) and row.p_boot_holm.iloc[0] < 0.05:
                v = auc[(auc.Model == model) & (auc.Class == cls)].AUC.iloc[0]
                ax.text(ci_ + (k - 3) * width, v + 0.06, "*", ha="center",
                        fontsize=11, fontweight="bold")
    ax.axhline(0.5, color=C_GREY, ls=":", lw=1)
    ax.set_xticks(x, classes, fontsize=9)
    ax.set_ylabel("One-vs-rest AUC", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7.2, frameon=False, loc="lower left", ncol=2)
    ax.set_title("(b) Per-class one-vs-rest AUC (95% CI, patient-level bootstrap)\n"
                 "* Holm-adjusted bootstrap p<0.05 vs Logistic Reg.", fontsize=10)
    save(fig, "Fig5_threeclass")


# ---------- Fig 6: deep-model learning process ----------
def fig6():
    h = pd.read_parquet("/mnt/shared-workspace/shared/jdim_r1/hist_binary_all.parquet")
    hh = h[h.epoch >= 0]
    deep = ["1D-CNN", "Hybrid CNN-Tr.", "Pure Transformer"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for k, model in enumerate(deep):
        ax = axes[k]
        g = hh[hh.model == model]
        # mean +/- SD across runs (seed x fold) per epoch
        agg = g.groupby("epoch").agg(
            tl=("train_loss", "mean"), tl_sd=("train_loss", "std"),
            vl=("val_loss", "mean"), vl_sd=("val_loss", "std"),
            ta=("train_auc", "mean"), va=("val_auc", "mean"),
            va_sd=("val_auc", "std"))
        ax2 = ax.twinx()
        ax.plot(agg.index, agg.tl, color=C_CLASSICAL, lw=1.8, label="train loss")
        ax.plot(agg.index, agg.vl, color=C_DEEP, lw=1.8, label="val loss")
        ax2.plot(agg.index, agg.ta, color=C_CLASSICAL, ls="--", lw=1.4,
                 label="train AUC")
        ax2.plot(agg.index, agg.va, color=C_DEEP, ls="--", lw=1.4,
                 label="val AUC")
        ax2.fill_between(agg.index, agg.va - agg.va_sd, agg.va + agg.va_sd,
                         color=C_DEEP, alpha=0.15, lw=0)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax2.set_ylabel("AUC", fontsize=10)
        ax2.set_ylim(0.4, 1.0)
        be = h[(h.model == model) & (h.epoch == -1)]["train_loss"]
        ax.axvline(be.median(), color=C_GREY, ls=":", lw=1.2)
        ax.text(be.median() + 1, ax.get_ylim()[1] * 0.95,
                f"median early stop = {int(be.median())}", fontsize=8,
                color="dimgrey")
        ax.set_title(f"{model}", fontsize=11)
        if k == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center right")
    fig.suptitle("Training vs grouped-validation learning curves (mean ± SD over "
                 "20 seeds × 5 folds; NIR-SC-UFES binary)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "Fig6_learning_curves")


# ---------- Supplementary figures ----------
def figS1():
    oof = pd.read_parquet("/mnt/shared-workspace/shared/jdim_r1/oof_binary_all.parquet")
    d = oof[oof.seed == 42]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for model in MODEL_ORDER:
        g = d[d.model == model]
        fpr, tpr, _ = roc_curve(g.y, g.p)
        from sklearn.metrics import auc as _auc
        col = C_CLASSICAL if model in MODEL_ORDER[:4] else C_DEEP
        ls = "-" if model in MODEL_ORDER[:4] else "--"
        ax.plot(fpr, tpr, ls, color=col, lw=1.6,
                alpha=1.0 - 0.13 * (MODEL_ORDER.index(model) % 4),
                label=f"{model} ({_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color=C_GREY, lw=1)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("ROC curves, pooled patient-level OOF (seed 42, n=647 spectra)",
                 fontsize=10.5)
    save(fig, "FigS1_roc")


def figS2():
    t4 = pd.read_csv(f"{TAB}/table4_external.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    t4 = t4.set_index("Model").loc[MODEL_ORDER].reset_index()
    cis = t4.AUC_CI.str.strip("[]").str.split(",", expand=True).astype(float)
    cols = [C_CLASSICAL if m in MODEL_ORDER[:4] else C_DEEP for m in t4.Model]
    ax.barh(range(7), t4.AUC, color=cols, alpha=0.85,
            xerr=[t4.AUC - cis[0], cis[1] - t4.AUC], capsize=3,
            error_kw=dict(lw=0.9, ecolor="black"))
    ax.axvline(0.5, color=C_GREY, ls="--", lw=1.2)
    ax.set_yticks(range(7), t4.Model, fontsize=9)
    ax.set_xlabel("ROC-AUC (95% CI)", fontsize=10)
    ax.set_xlim(0.2, 0.9)
    ax.set_title("External validation on Deteccthia (n=152 photo-level means;\n"
                 "models trained on NIR-SC-UFES, applied without re-fitting)",
                 fontsize=10)
    save(fig, "FigS2_external")


def figS3():
    oof = pd.read_parquet("/mnt/shared-workspace/shared/jdim_r1/oof_binary_all.parquet")
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for (seed, model), g in oof.groupby(["seed", "model"]):
        aucs.setdefault(model, []).append(roc_auc_score(g.y, g.p))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    data = [aucs[m] for m in MODEL_ORDER]
    bp = ax.boxplot(data, tick_labels=MODEL_ORDER, patch_artist=True, widths=0.55)
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(C_CLASSICAL if m in MODEL_ORDER[:4] else C_DEEP)
        patch.set_alpha(0.55)
    for i, m in enumerate(MODEL_ORDER):
        ax.scatter(np.full(len(aucs[m]), i + 1) + np.random.default_rng(0).normal(0, 0.05, len(aucs[m])),
                   aucs[m], s=14, color="black", alpha=0.6, zorder=3)
    ax.set_ylabel("Pooled out-of-fold AUC", fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Repeated 5-fold CV across 20 random seeds (patient-level groups)",
                 fontsize=10.5)
    save(fig, "FigS3_seeds")


def figS4():
    oof = pd.read_parquet("/mnt/shared-workspace/shared/jdim_r1/oof_binary_all.parquet")
    d = oof[(oof.seed == 42) & (oof.model == "Logistic Reg.")]
    y = d.y.to_numpy(); p_raw = d.p.to_numpy()
    p_cal = np.load("/workspace/benchmark/out/p_cal_seed42.npy")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for name, pp, col, mk in [("Raw", p_raw, C_DEEP, "o"),
                              ("Platt-scaled (cross-fitted)", p_cal, C_CLASSICAL, "s")]:
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pp, bins) - 1, 0, 9)
        mp, mo, nn = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                mp.append(pp[m].mean()); mo.append(y[m].mean()); nn.append(m.sum())
        ax.plot([0, 1], [0, 1], ":", color=C_GREY, lw=1)
        ax.scatter(mp, mo, marker=mk, s=[max(20, n / 3) for n in nn],
                   color=col, alpha=0.85, label=name, zorder=3)
        ax.plot(mp, mo, color=col, lw=1.2, alpha=0.6)
    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Observed malignancy frequency", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("Reliability diagram — Logistic Reg. (seed 42 OOF)\n"
                 "ECE 0.077 raw → 0.038 Platt; Brier 0.152 → 0.150", fontsize=10)
    save(fig, "FigS4_calibration")


def figS5():
    ab = pd.read_csv("/workspace/benchmark/out/ablation.csv")
    chains = ["raw", "l2", "l2_snv", "l2_snv_sg"]
    models = ["Logistic Reg.", "1D-CNN", "Pure Transformer"]
    labels = {"raw": "raw", "l2": "L2", "l2_snv": "L2+SNV",
              "l2_snv_sg": "L2+SNV+SG-deriv"}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(4); width = 0.26
    cols = {"Logistic Reg.": C_CLASSICAL, "1D-CNN": C_DEEP,
            "Pure Transformer": "#009E73"}
    for k, m in enumerate(models):
        g = ab[ab.model == m].groupby("chain").AUC
        means = [g.mean()[c] for c in chains]
        sds = [g.std()[c] for c in chains]
        ax.bar(x + (k - 1) * width, means, width * 0.9, yerr=sds, capsize=3,
               color=cols[m], label=m, error_kw=dict(lw=0.8, ecolor="black"))
    ax.set_xticks(x, [labels[c] for c in chains], fontsize=9)
    ax.set_ylabel("Pooled OOF AUC (mean ± SD, seeds 42–44)", fontsize=9.5)
    ax.set_ylim(0.5, 0.85)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("Preprocessing-chain sensitivity (patient-level grouped CV)",
                 fontsize=10.5)
    save(fig, "FigS5_ablation")


if __name__ == "__main__":
    fig3(); fig4(); fig5(); fig6(); figS1(); figS2(); figS3(); figS4(); figS5()
    print("all figures done")
