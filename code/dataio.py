"""Data loading and task construction for the JDIM two-cohort NIR benchmark.

Datasets (both public):
- NIR-SC-UFES (Mendeley j9773cyr3k, v1, 2024-01-18): 714 spectra, 125 channels
  (908.1-1676.2 nm), column `Sample` = "<patient>_<replicate>" -> patient groups.
- Deteccthia (github.com/LACourtenay/Deteccthia_Skin_Cancer_Project, commit 65b78f8):
  8250 spectra x 150 bands; `Photo_ID` groups replicate spectra of one lesion/photo.

All grouping for CV is at the PATIENT (UFES) / PHOTO (Deteccthia) level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UFES_PATH = "/mnt/shared-workspace/shared/jdim_r1/nir_sc_ufes_raw.csv"
DET_PATH = "/mnt/shared-workspace/shared/jdim_r1/deteccthia_raw.csv"

MAL_CLASSES = {"ACK", "CBC", "CEC", "MEL"}   # pre-malignant + malignant
BEN_CLASSES = {"NEV", "SEK"}
THREE_CLASS_MAP = {"NEV": "benign", "SEK": "benign",
                   "ACK": "pre-malignant (AK)",
                   "CBC": "carcinoma (BCC/SCC)", "CEC": "carcinoma (BCC/SCC)"}


def load_ufes(path: str = UFES_PATH):
    """Return X (714x125 float), wavelengths (125,), meta DataFrame."""
    df = pd.read_csv(path)
    wl_cols = [c for c in df.columns if c not in ("N", "Sample", "Class", "yClass", "patient")]
    wl = np.array([float(c) for c in wl_cols])
    X = df[wl_cols].to_numpy(dtype=np.float64)
    meta = df[["N", "Sample", "Class", "patient"]].copy()
    return X, wl, meta


def ufes_binary(path: str = UFES_PATH):
    """Binary task: malignant/pre-malignant {ACK,CBC,CEC,MEL} vs benign {NEV,SEK}.

    Returns X, y (1=malignant), groups (patient), meta subset. n=647 spectra.
    """
    X, wl, meta = load_ufes(path)
    keep = meta["Class"].isin(MAL_CLASSES | BEN_CLASSES)
    X, meta = X[keep.to_numpy()], meta[keep].reset_index(drop=True)
    y = meta["Class"].isin(MAL_CLASSES).astype(int).to_numpy()
    groups = meta["patient"].to_numpy()
    return X, y, groups, wl, meta


def ufes_threeclass(path: str = UFES_PATH):
    """Three-class task: benign {NEV,SEK} vs pre-malignant {ACK} vs carcinoma {CBC,CEC}.

    MEL (6 spectra) excluded as too few. n=641 spectra.
    """
    X, wl, meta = load_ufes(path)
    keep = meta["Class"].isin(THREE_CLASS_MAP)
    X, meta = X[keep.to_numpy()], meta[keep].reset_index(drop=True)
    labels = sorted(set(THREE_CLASS_MAP.values()))  # benign, carcinoma, pre-malignant
    order = ["benign", "pre-malignant (AK)", "carcinoma (BCC/SCC)"]
    y = np.array([order.index(THREE_CLASS_MAP[c]) for c in meta["Class"]])
    groups = meta["patient"].to_numpy()
    return X, y, groups, wl, meta, order


def load_deteccthia(path: str = DET_PATH):
    """Return photo-level mean spectra DataFrame: X (n_photos x 150), y, photo ids, band nm."""
    df = pd.read_csv(path)
    band_cols = [c for c in df.columns if c.startswith("BAND_")]
    freq = pd.read_csv("/workspace/data/deteccthia/Robust_Statistics/Camera_Frequency_Details.csv")
    # camera file: try to find wavelength column
    nm = None
    for c in freq.columns:
        vals = pd.to_numeric(freq[c], errors="coerce")
        if vals.notna().sum() >= 150 and vals.min() < 1000 and vals.max() > 1500:
            nm = vals.to_numpy()[:150]
            break
    if nm is None:  # fallback: linear axis 900-1600 nm (documented assumption)
        nm = np.linspace(900, 1600, 150)
    g = df.groupby("Photo_ID")
    X = g[band_cols].mean().to_numpy(dtype=np.float64)
    cls = g["Sample"].first().to_numpy()
    y = np.isin(cls, ["AK", "SCC", "BCC"]).astype(int)  # 1=malignant, 0=healthy
    photos = np.array(sorted(df["Photo_ID"].unique()))
    return X, y, photos, nm, cls


def harmonize_grid(X_src, nm_src, nm_dst):
    """Linearly interpolate spectra from source wavelength axis onto destination axis.

    Only destination points inside the source range are valid; caller must
    restrict both to the common range first.
    """
    out = np.empty((X_src.shape[0], len(nm_dst)))
    for i, row in enumerate(X_src):
        out[i] = np.interp(nm_dst, nm_src, row)
    return out


def external_pair():
    """Build (X_train, y_train, groups) from UFES and (X_ext, y_ext) from Deteccthia
    on the common wavelength range, Deteccthia interpolated onto the UFES grid."""
    X_u, y_u, g_u, wl_u, _ = ufes_binary()
    X_d, y_d, photos, nm_d, _ = load_deteccthia()
    lo, hi = max(wl_u.min(), nm_d.min()), min(wl_u.max(), nm_d.max())
    keep_u = (wl_u >= lo) & (wl_u <= hi)
    wl_common = wl_u[keep_u]
    X_u_c = X_u[:, keep_u]
    X_d_c = harmonize_grid(X_d, nm_d, wl_common)
    return X_u_c, y_u, g_u, X_d_c, y_d, wl_common, photos
