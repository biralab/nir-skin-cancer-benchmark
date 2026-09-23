"""Deep spectral models (PyTorch, CPU) with a fully logged training loop.

Architectures are deliberately small (small-cohort regime) and fully documented
for the revised manuscript (Reviewer 1, comment 5):

- CNN1D:   Conv1d(1,32,k=9,s=2) -> BN -> ReLU -> Conv1d(32,64,k=7,s=2) -> BN ->
           ReLU -> Conv1d(64,128,k=5,s=2) -> BN -> ReLU -> GAP -> Dropout(0.3)
           -> Linear(128,n_classes).            (~83k parameters, p=125)
- Hybrid:  conv stem Conv1d(1,32,k=9,s=2)+Conv1d(32,64,k=5,s=2) -> linear proj to
           d_model=64 + learned positional embedding -> TransformerEncoder
           (2 layers, nhead=4, dim_ff=128, dropout=0.1, GELU) -> mean-pool ->
           Linear(64,n_classes).                (~60k parameters)
- Transformer: patch embedding (patch=5 -> 25 tokens, linear 5->64) + learned
           positional embedding + CLS token -> TransformerEncoder (2 layers,
           nhead=4, dim_ff=128, dropout=0.1, GELU) -> CLS head. (~50k parameters)

Training: AdamW (lr 1e-3, weight_decay 1e-4), batch 32, cross-entropy with
inverse-frequency class weights, <=200 epochs, early stopping (patience 20,
min_delta 1e-4) on an inner grouped validation split (80/20 by patient) carved
from the training fold only. Per-epoch train/val loss and AUC are logged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DEEP = ["1D-CNN", "Hybrid CNN-Tr.", "Pure Transformer"]


class CNN1D(nn.Module):
    def __init__(self, n_ch=125, n_classes=2, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


class HybridCNNTransformer(nn.Module):
    def __init__(self, n_ch=125, n_classes=2, d_model=64, nhead=4, layers=2,
                 dim_ff=128, p_drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
        )
        n_tok = int(np.ceil(int(np.ceil(n_ch / 2)) / 2))
        self.proj = nn.Linear(64, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, p_drop,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(d_model, n_classes))

    def forward(self, x):
        z = self.stem(x.unsqueeze(1)).transpose(1, 2)   # (B, T, 64)
        z = self.proj(z) + self.pos
        z = self.encoder(z).mean(dim=1)
        return self.head(z)


class PureTransformer(nn.Module):
    def __init__(self, n_ch=125, n_classes=2, patch=5, d_model=64, nhead=4,
                 layers=2, dim_ff=128, p_drop=0.1):
        super().__init__()
        self.patch = patch
        self.n_tok = n_ch // patch
        self.embed = nn.Linear(patch, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_tok + 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, p_drop,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(d_model, n_classes))

    def forward(self, x):
        B = x.shape[0]
        z = x[:, : self.n_tok * self.patch].reshape(B, self.n_tok, self.patch)
        z = self.embed(z)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1) + self.pos
        z = self.encoder(z)
        return self.head(z[:, 0])


def make_deep(name, n_ch=125, n_classes=2):
    if name == "1D-CNN":
        return CNN1D(n_ch, n_classes)
    if name == "Hybrid CNN-Tr.":
        return HybridCNNTransformer(n_ch, n_classes)
    if name == "Pure Transformer":
        return PureTransformer(n_ch, n_classes)
    raise ValueError(name)


def count_params(name, n_ch=125, n_classes=2):
    return sum(p.numel() for p in make_deep(name, n_ch, n_classes).parameters())


def train_deep(name, Xtr, ytr, gtr, seed=0, max_epochs=200, patience=20,
               batch=32, lr=1e-3, wd=1e-4, log_path=None, inner_seed=None):
    """Train one deep model with grouped inner-validation early stopping.

    Returns (predict_callable, history dict). history holds per-epoch
    train/val loss and AUC plus the early-stopped epoch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed if inner_seed is None else inner_seed)
    n_classes = int(np.max(ytr)) + 1
    # inner grouped split for early stopping (from the training fold only)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(Xtr, ytr, gtr))
    Xtr_i, Xva_i = Xtr[tr_idx], Xtr[va_idx]
    ytr_i, yva_i = ytr[tr_idx], ytr[va_idx]

    model = make_deep(name, Xtr.shape[1], n_classes)
    counts = np.bincount(ytr_i, minlength=n_classes).astype(np.float64)
    w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                     dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Xtr_i, dtype=torch.float32)
    ytr_t = torch.tensor(ytr_i, dtype=torch.long)
    Xva_t = torch.tensor(Xva_i, dtype=torch.float32)
    yva_t = torch.tensor(yva_i, dtype=torch.long)

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_auc": [],
            "val_auc": []}
    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    n = len(Xtr_t)
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr_t[j])
            loss = lossf(out, ytr_t[j])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(j)
        ep_loss /= n
        model.eval()
        with torch.no_grad():
            ptr = torch.softmax(model(Xtr_t), 1)[:, -1].numpy()
            pva = torch.softmax(model(Xva_t), 1)[:, -1].numpy()
            vloss = lossf(model(Xva_t), yva_t).item()
        def _auc(yy, pp):
            try:
                return roc_auc_score(yy, pp) if n_classes == 2 else \
                    roc_auc_score(yy, pp, multi_class="ovr")
            except ValueError:
                return np.nan
        hist["epoch"].append(epoch)
        hist["train_loss"].append(ep_loss)
        hist["val_loss"].append(vloss)
        hist["train_auc"].append(_auc(ytr_i, ptr))
        hist["val_auc"].append(_auc(yva_i, pva))
        if vloss < best_val - 1e-4:
            best_val, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["n_epochs_run"] = len(hist["epoch"])
    hist["best_val_loss"] = best_val

    def predict_proba(X):
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()

    return predict_proba, hist
