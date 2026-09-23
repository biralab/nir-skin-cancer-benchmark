# -*- coding: utf-8 -*-
"""Rebuild Fig. 2 (evidence landscape) with literature-verified numbers only.
Sources: /mnt/shared-workspace/shared/jdim_r1/literature_verification.md
Metric types: AUC (circle) vs accuracy/balanced-accuracy as reported (triangle).
Courtenay 2024 [28] excluded (no classifier metric reported); Varga 2025 [30]
excluded (meta-analysis, no single cohort)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#999999"

# (n, value, metric('auc'|'acc'), family('C'|'D'|'M'), label, dx, dy)
pts = [
    (436,   0.98,  "auc", "C", "[61]",  1.15,  0.012),   # Araujo 2021 LightGBM
    (617,   0.75,  "auc", "C", "[13]",  1.15, -0.028),   # Bratchenko 2021 PLS-DA
    (731,   0.870, "auc", "C", "[19]",  0.62,  0.014),   # Zhao 2024 PLS-DA
    (971,   0.839, "acc", "C", "[27]",  1.15,  0.012),   # Loss 2024 LightGBM bal-acc
    (149,   0.766, "acc", "C", "[25]",  1.15, -0.030),   # Wang 2024 LR (PCA)
    (109,   0.776, "auc", "M", "[29]",  1.15,  0.010),   # Dragka 2026 NIRIMP (mixed)
    (617,   0.96,  "auc", "D", "[18]",  1.15, -0.032),   # Bratchenko 2022 CNN
    (731,   0.886, "auc", "D", "[19]",  1.18,  0.010),   # Zhao 2024 CNN
    (20000, 0.98,  "acc", "D", "[26]",  0.55, -0.030),   # Qiu 2023 CNN SERS
    (149,   0.9254,"auc", "D", "[25]",  1.15,  0.012),   # Wang 2024 TCLP
    (1200,  0.968, "acc", "D", "[62]",  1.15, -0.030),   # ChuDuc 2025 hybrid
    (100,   0.86,  "auc", "D", "[10]",  1.15,  0.012),   # Haenssle 2018 (images, open)
]
ours = [
    (647, 0.774, "C", "This work\n(LogReg)"),
    (647, 0.663, "D", "This work\n(1D-CNN)"),
]
fc = {"C": BLUE, "D": ORANGE, "M": GRAY}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for n, v, met, fam, lab, dx, dy in pts:
    open_m = (lab == "[10]")
    ax.scatter(n, v, marker="o" if met == "auc" else "^", s=64,
               facecolors="none" if open_m else fc[fam], edgecolors=fc[fam],
               linewidths=1.4, zorder=3)
    ax.annotate(lab, (n, v), xytext=(n * dx if dx > 0.8 else n * dx, v + dy),
                fontsize=8, color="#333333", zorder=4)
for n, v, fam, lab in ours:
    ax.scatter(n, v, marker="*", s=300, c=fc[fam], edgecolors="black",
               linewidths=0.8, zorder=5)
    ax.annotate(lab, (n, v), xytext=(n * 1.28, v - 0.006), fontsize=8.5,
                fontweight="bold", color=fc[fam], zorder=5)

ax.set_xscale("log")
ax.set_xlim(80, 40000)
ax.set_ylim(0.55, 1.03)
ax.set_xlabel("Cohort size (spectra / lesions / images, log scale)")
ax.set_ylabel("Reported discrimination")
ax.grid(True, which="both", lw=0.3, alpha=0.5)
ax.axvspan(80, 1000, color="#f0f0f0", zorder=0)
ax.text(95, 0.575, "sub-1000 regime", fontsize=8, color="#666666")

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=8, label="Classical"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=8, label="Deep"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markersize=8, label="Mixed / not attributable"),
    Line2D([0], [0], marker="o", color="#555555", markersize=8, label="ROC AUC"),
    Line2D([0], [0], marker="^", color="#555555", markersize=8, label="Accuracy (as reported)"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#555555", markeredgecolor="black", markersize=13, label="This work (Table 3)"),
]
ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig("/workspace/benchmark/figures/Fig2_landscape.png", dpi=300)
fig.savefig("/workspace/benchmark/figures/Fig2_landscape.svg")
print("saved Fig2_landscape.png/.svg")
