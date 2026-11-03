"""Training, evaluation and comparison plots."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from . import plotstyle as ps
from .config import COST_FN, COST_FP, CV_FOLDS, FIG_DIR, RANDOM_STATE
from .models import TUNE_GRIDS
from .preprocessing import build_pipeline

warnings.filterwarnings("ignore")


def fit_and_evaluate(models, X_tr, X_te, y_tr, y_te, k=10, tune=False, drop=()):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, probas = [], {}, {}
    for name, model in models.items():
        pipe = build_pipeline(model, k=k, drop=drop)
        if tune and name in TUNE_GRIDS:
            gs = GridSearchCV(pipe, TUNE_GRIDS[name], cv=cv, scoring="accuracy", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_
            cv_mean, cv_std = gs.best_score_, 0.0
        else:
            scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_mean, cv_std = scores.mean(), scores.std()
            pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        rows.append({
            "Model": name, "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, proba),
            "CV Accuracy": cv_mean, "CV Std": cv_std,
            "Business Cost": COST_FN * fn + COST_FP * fp,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
        fitted[name], probas[name] = pipe, proba
    table = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    table.index += 1
    return table, fitted, probas


def plot_comparison(table, baseline, fname="10_model_comparison.png", title="Model comparison (test set)"):
    t = table.sort_values("Accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    cols = [ps.GOOD if m == table.iloc[0]["Model"] else ps.ACC for m in t["Model"]]
    ax.barh(t["Model"], t["Accuracy"] * 100, color=cols)
    ax.axvline(baseline * 100, color=ps.WARN, ls="--", label=f"majority-class baseline {baseline*100:.0f}%")
    for y, v in enumerate(t["Accuracy"] * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    ax.set_xlim(60, 90)
    ax.set_title("Test accuracy (%)")
    ax.legend(loc="lower right")
    ax = axes[1]
    m = t.set_index("Model")[["Precision", "Recall", "F1", "ROC-AUC"]]
    m.plot.barh(ax=ax, width=0.8, color=[ps.ACC, ps.BAD, ps.WARN, ps.GOOD])
    ax.set_xlim(0, 1)
    ax.set_title("Precision / Recall / F1 / ROC-AUC (default = positive class)")
    ax.set_ylabel("")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def plot_roc(probas, y_te, fname="11_roc_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, len(probas)))
    for (name, p), col in zip(probas.items(), palette):
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=ps.GRID)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - all models")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def plot_confusions(fitted, X_te, y_te, fname="12_confusion_matrices.png"):
    n = len(fitted)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    for ax, (name, pipe) in zip(axes.ravel(), fitted.items()):
        cm = confusion_matrix(y_te, pipe.predict(X_te))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if v > cm.max() / 2 else ps.BG)
        ax.set_xticks([0, 1], ["Good", "Default"])
        ax.set_yticks([0, 1], ["Good", "Default"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def repeated_cv_comparison(models, X, y, k=10, drop=(), repeats=3):
    """More reliable than one 200-row test split: repeated stratified CV on all 1000 loans."""
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=repeats, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        res = cross_validate(build_pipeline(model, k=k, drop=drop), X, y, cv=cv,
                             scoring=["accuracy", "roc_auc", "recall", "f1"], n_jobs=1)
        rows.append({"Model": name,
                     "Accuracy": res["test_accuracy"].mean(), "Accuracy Std": res["test_accuracy"].std(),
                     "ROC-AUC": res["test_roc_auc"].mean(), "Recall": res["test_recall"].mean(),
                     "F1": res["test_f1"].mean()})
    out = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


def plot_repeated_cv(out, baseline, fname="16_repeated_cv_accuracy.png", title_suffix=""):
    t = out.sort_values("Accuracy")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(t["Model"], t["Accuracy"] * 100, xerr=t["Accuracy Std"] * 100,
            color=[ps.GOOD if m == out.iloc[0]["Model"] else ps.ACC for m in t["Model"]],
            error_kw={"ecolor": ps.TEXT, "capsize": 3})
    ax.axvline(baseline * 100, color=ps.WARN, ls="--", label=f"baseline {baseline*100:.0f}%")
    ax.set_xlim(60, 84)
    ax.set_title("Repeated 5-fold CV accuracy, all 1000 loans (mean ± std)" + title_suffix)
    ax.set_xlabel("Accuracy (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
