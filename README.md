# Code Snapshot S1 — JDIM-D-26-02857 (R1)

Reproduction package for "Optical Spectroscopy and Machine Learning for
Skin-Cancer Detection: Why Cohort Size, Validation Rigor and Calibration —
Not Architecture — Govern Clinical Readiness" (Journal of Imaging
Informatics in Medicine, major revision R1).

Repository: https://github.com/biralab/nir-skin-cancer-benchmark
A DOI-minted archive (Zenodo) will be linked upon acceptance.

## Data (not included — public sources)
- NIR-SC-UFES: Mendeley Data, doi:10.17632/j9773cyr3k.1 (714 spectra / 331
  patients; binary task: 647 spectra / 319 patients).
- Deteccthia: github.com/LACourtenay/Deteccthia_Skin_Cancer_Project
  (commit 65b78f8; 152 photo-level mean spectra: 115 malignant, 37 healthy).

Place the raw CSVs as `nir_sc_ufes_raw.csv` and `deteccthia_raw.csv` in the
working directory; `code/dataio.py` documents the expected schema.

## Layout
- `code/` — pipeline: dataio, preprocess (fold-isolated chains), models
  (classical + deep), runners (full 20-seed CV, nested CV, external,
  three-class, ablation), statistics and figure scripts, manuscript engine.
- `tables/` — every number in main-text Tables 3–7, S1–S5 and figure panels.
- `predictions/` — pooled out-of-fold predictions (all 20 seeds, binary and
  three-class), per-epoch training logs, external-validation predictions
  (standard and re-standardised), nested-CV selections, ablation results.
  Fold definitions are recoverable from the (seed, fold, idx, group) columns.
- `figures/` — all main and supplementary figures (PNG + SVG).

## Reproduce
Python 3.11, packages: numpy, pandas, scikit-learn, scipy, torch,
python-docx, matplotlib. Run order: `run_full.py` (20 seeds) ->
`run_nested.py` -> `run_external.py` -> `run_external_restd.py` ->
`run_threeclass.py` -> `run_ablation.py` -> `compute_tables.py` ->
`compute_threeclass.py` -> `redo_delong.py` -> `make_figures.py` /
`make_fig2.py`. All seeds are fixed (42-61; pilot seed 42).
