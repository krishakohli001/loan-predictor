"""Leak-free preprocessing: everything is fitted on training data only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CAP_COLS, CAT_COLS, NUM_COLS


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap outliers at the IQR fences: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""

    def __init__(self, cols=None, factor=1.5):
        self.cols = cols
        self.factor = factor

    def fit(self, X, y=None):
        self.fences_ = {}
        for c in (self.cols or []):
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.fences_[c] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.fences_.items():
            X[c] = X[c].clip(lo, hi)
        return X


def build_pipeline(model, k=10, drop=()):
    """IQR capping -> scale/one-hot -> SelectKBest (ANOVA F) -> model."""
    num = [c for c in NUM_COLS if c not in drop]
    cat = [c for c in CAT_COLS if c not in drop]
    cap = [c for c in CAP_COLS if c not in drop]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat),
    ])
    return Pipeline([
        ("cap", IQRCapper(cols=cap)),
        ("prep", prep),
        ("select", SelectKBest(f_classif, k=k)),
        ("model", model),
    ])


def selected_feature_names(pipe):
    """Names of the features that survived SelectKBest (prefixes stripped)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    mask = pipe.named_steps["select"].get_support()
    return [n.split("__", 1)[1] for n in np.asarray(names)[mask]]


def transform_selected(pipe, X):
    """Run X through every step except the final model."""
    return pipe[:-1].transform(X)
