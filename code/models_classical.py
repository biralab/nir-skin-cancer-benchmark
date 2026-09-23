"""Classical chemometric models (fixed, documented hyper-parameters for the
main runs; nested CV estimates tuning optimism separately)."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Fixed hyper-parameters (majority choice of the nested-CV audit, Table 6)
FIXED_HP = {
    "Logistic Reg.": {"C": 10.0, "max_iter": 5000},
    "PLS-DA+SVM": {"n_components": 20},
    "PCA-LDA": {"n_components": 30},
    "Random Forest": {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2},
}

HP_GRIDS = {
    "Logistic Reg.": {"C": [0.01, 0.1, 1, 10, 100]},
    "PLS-DA+SVM": {"n_components": [5, 10, 15, 20, 25, 30]},
    "PCA-LDA": {"n_components": [5, 10, 15, 20, 25, 30]},
    "Random Forest": {"max_depth": [4, 8, 16, None]},
}

CLASSICAL = list(FIXED_HP.keys())


class PLSDASVM:
    """PLS2 projection onto one-hot labels, then RBF SVM on the scores."""

    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components)
        Y = np.column_stack([1 - y, y])
        self.pls_.fit(X, Y)
        T = self.pls_.x_scores_
        self.svm_ = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                        random_state=0)
        self.svm_.fit(T, y)
        return self

    def predict_proba(self, X):
        T = self.pls_.transform(X)
        return self.svm_.predict_proba(T)


class PCALDA:
    def __init__(self, n_components=30):
        self.n_components = n_components

    def fit(self, X, y):
        self.pca_ = PCA(n_components=self.n_components)
        Z = self.pca_.fit_transform(X)
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(Z, y)
        return self

    def predict_proba(self, X):
        return self.lda_.predict_proba(self.pca_.transform(X))


def make_classical(name, hp=None):
    hp = dict(FIXED_HP[name]) if hp is None else hp
    if name == "Logistic Reg.":
        return LogisticRegression(penalty="l2", solver="lbfgs", **hp)
    if name == "PLS-DA+SVM":
        return PLSDASVM(**hp)
    if name == "PCA-LDA":
        return PCALDA(**hp)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=0, n_jobs=1, **hp)
    raise ValueError(name)
