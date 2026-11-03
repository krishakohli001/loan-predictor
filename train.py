"""Run the complete pipeline:  python train.py [--tune] [--drop-sensitive]

Outputs: figures + CSV tables in reports/, trained models in models/.
"""
import argparse
import json
import platform

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src import plotstyle as ps
from src.config import (FIG_DIR, MODEL_DIR, RANDOM_STATE, RES_DIR, SENSITIVE_COLS, TARGET,
                        TEST_SIZE, TOP_K)
from src.data import add_features, load_raw, prepare
from src.eda import run_eda
from src.evaluate import (fit_and_evaluate, plot_comparison, plot_confusions, plot_repeated_cv,
                          plot_roc, repeated_cv_comparison)
from src.models import get_models, slug
from src.preprocessing import selected_feature_names
from src.stats_model import odds_ratio_table, plot_odds_ratios


def run(tune=False, drop_sensitive=False, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    for d in (FIG_DIR, RES_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    drop = tuple(SENSITIVE_COLS) if drop_sensitive else ()

    log("1-5) Cleaning, outliers, EDA, visualisation, feature engineering ...")
    df = add_features(prepare(load_raw()))
    run_eda(df)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
    baseline = 1 - y_te.mean()
    log(f"   train={len(X_tr)}  test={len(X_te)}  default rate={y.mean():.1%}  baseline acc={baseline:.1%}")

    log(f"6-7,9) Training 9 models on the top-{TOP_K} ANOVA-F features ...")
    table, fitted, probas = fit_and_evaluate(get_models(), X_tr, X_te, y_tr, y_te, k=TOP_K,
                                             tune=tune, drop=drop)
    table.to_csv(RES_DIR / "model_comparison_top10.csv", index_label="Rank")
    plot_comparison(table, baseline)
    plot_roc(probas, y_te)
    plot_confusions(fitted, X_te, y_te)

    log("   Extra experiment: all features (no SelectKBest) ...")
    table_all, _, _ = fit_and_evaluate(get_models(), X_tr, X_te, y_tr, y_te, k="all", tune=tune, drop=drop)
    table_all.to_csv(RES_DIR / "model_comparison_all_features.csv", index_label="Rank")
    plot_comparison(table_all, baseline, "14_model_comparison_all_features.png",
                    "Model comparison - ALL features (no feature selection)")

    log("   Robust check: repeated 5-fold CV on all 1000 loans ...")
    rcv = repeated_cv_comparison(get_models(), X, y, k=TOP_K, drop=drop)
    rcv.to_csv(RES_DIR / "repeated_cv_top10.csv", index_label="Rank")
    plot_repeated_cv(rcv, 1 - y.mean())
    rcv_all = repeated_cv_comparison(get_models(), X, y, k="all", drop=drop)
    rcv_all.to_csv(RES_DIR / "repeated_cv_all_features.csv", index_label="Rank")
    plot_repeated_cv(rcv_all, 1 - y.mean(), "17_repeated_cv_all_features.png", " - ALL features")

    log("8) Coefficient estimation by maximum likelihood (statsmodels) ...")
    lr_pipe = fitted["Logistic Regression"]
    odds, _ = odds_ratio_table(lr_pipe, X_tr, y_tr)
    plot_odds_ratios(odds)

    # Random-forest importance plot
    rf = fitted["Random Forest"]
    names = selected_feature_names(rf)
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color=ps.ACC)
    ax.set_title("Random Forest feature importance (top-10 selected features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_rf_feature_importance.png")
    plt.close(fig)

    # Persist models + metadata for the app
    for name, pipe in fitted.items():
        joblib.dump(pipe, MODEL_DIR / f"{slug(name)}.joblib", compress=3)
    best = table.iloc[0]
    meta = {
        "best_model": best["Model"], "best_accuracy": float(best["Accuracy"]),
        "best_model_repeated_cv": rcv.iloc[0]["Model"], "best_model_repeated_cv_all": rcv_all.iloc[0]["Model"], "best_repeated_cv_accuracy": float(rcv.iloc[0]["Accuracy"]),
        "baseline_accuracy": float(baseline), "selected_features": selected_feature_names(lr_pipe),
        "sklearn_version": sklearn.__version__, "python": platform.python_version(),
        "tuned": tune, "dropped_sensitive": drop_sensitive,
        "models": list(fitted.keys()),
    }
    (RES_DIR / "summary.json").write_text(json.dumps(meta, indent=2))
    log("\n=== Top-10-feature comparison (test set) ===")
    log(table[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Accuracy", "Business Cost"]]
        .round(3).to_string())
    log("\n=== Repeated CV (all 1000 loans) ===")
    log(rcv.round(3).to_string())
    log("\n=== Repeated CV, ALL features ===")
    log(rcv_all.round(3).to_string())
    log(f"\nBEST MODEL: {best['Model']}  (accuracy {best['Accuracy']:.1%} vs baseline {baseline:.1%})")
    return table, table_all, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid-search hyper-parameters (slower)")
    ap.add_argument("--drop-sensitive", action="store_true",
                    help="exclude sex/marital status, foreign-worker and age (fairness check)")
    a = ap.parse_args()
    run(tune=a.tune, drop_sensitive=a.drop_sensitive)
