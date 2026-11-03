"""Exploratory data analysis: statistics, outliers and the five required chart types."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import plotstyle as ps
from .config import CAP_COLS, FIG_DIR, RES_DIR, TARGET
from .preprocessing import IQRCapper


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, mode, variance, std and z-score summary for numeric columns."""
    num = df.select_dtypes("number").drop(columns=[TARGET])
    z = num.apply(stats.zscore)
    out = pd.DataFrame({
        "mean": num.mean(), "median": num.median(), "mode": num.mode().iloc[0],
        "variance": num.var(), "std": num.std(), "min": num.min(), "max": num.max(),
        "skewness": num.skew(), "max_|z|": z.abs().max(), "n_|z|>3": (z.abs() > 3).sum(),
    }).round(3)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES_DIR / "descriptive_statistics.csv")
    return out


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    cap = IQRCapper(cols=CAP_COLS).fit(df)
    rows = []
    for c, (lo, hi) in cap.fences_.items():
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"feature": c, "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
                     "outliers_capped": n})
    rep = pd.DataFrame(rows)
    rep.to_csv(RES_DIR / "outlier_report.csv", index=False)
    return rep


def run_eda(df: pd.DataFrame):
    """`df` must already contain engineered features (monthly_burden, age_group)."""
    ps.apply()
    desc = descriptive_stats(df)
    rep = outlier_report(df)
    capped = IQRCapper(cols=CAP_COLS).fit(df).transform(df)
    lab = df[TARGET].map({0: "Good (0)", 1: "Default (1)"})
    pal = {"Good (0)": ps.GOOD, "Default (1)": ps.BAD}

    # 1) Pie chart - class balance
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df[TARGET].value_counts().sort_index()
    ax.pie(counts, labels=["Good risk", "Default"], colors=[ps.GOOD, ps.BAD], autopct="%1.1f%%",
           startangle=90, wedgeprops={"edgecolor": ps.PANEL, "linewidth": 2},
           textprops={"color": ps.BG, "fontweight": "bold"})
    ax.set_title("Class balance of Risk")
    ax.grid(False)
    _save(fig, "01_pie_risk_distribution.png")

    # 2) Bar charts - default rate by category
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), ["checking_balance", "credit_history", "purpose", "savings_balance"]):
        rate = df.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
        ax.barh(rate.index, rate.values, color=ps.ACC)
        ax.axvline(df[TARGET].mean() * 100, color=ps.WARN, ls="--", label="overall 30%")
        ax.set_title(f"Default rate (%) by {col}")
        ax.invert_yaxis()
        ax.legend(loc="lower right")
    _save(fig, "02_bar_default_rate_by_category.png")

    # 3) Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), ["duration_months", "credit_amount", "age", "monthly_burden"]):
        sns.histplot(data=df.assign(Risk_label=lab), x=col, hue="Risk_label", palette=pal, kde=True,
                     bins=30, ax=ax, element="step", stat="density", common_norm=False)
        ax.set_title(f"Distribution of {col}")
    _save(fig, "03_histograms.png")

    # 4) Box plots before / after capping
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for j, col in enumerate(CAP_COLS):
        for i, (d, tag) in enumerate([(df, "before"), (capped, "after")]):
            sns.boxplot(y=d[col], ax=axes[i, j], color=ps.ACC, flierprops={"markerfacecolor": ps.BAD,
                        "markeredgecolor": ps.BAD}, medianprops={"color": ps.WARN})
            axes[i, j].set_title(f"{col} ({tag} capping)")
    _save(fig, "04_boxplots_outliers.png")

    # 5) Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    ax.set_title("Correlation heatmap (numeric features + Risk)")
    ax.grid(False)
    _save(fig, "05_heatmap_correlation.png")

    # 6) Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df.assign(Risk_label=lab), x="duration_months", y="credit_amount",
                    hue="Risk_label", palette=pal, alpha=0.7, ax=ax)
    ax.set_title("Credit amount vs duration (colored by Risk)")
    _save(fig, "06_scatter_amount_vs_duration.png")

    # 7) z-score plot
    fig, ax = plt.subplots(figsize=(8, 5))
    z = stats.zscore(df["credit_amount"])
    ax.hist(z, bins=40, color=ps.ACC)
    for s in (-3, 3):
        ax.axvline(s, color=ps.BAD, ls="--")
    ax.set_title(f"z-scores of credit_amount ({int((abs(z) > 3).sum())} points with |z| > 3)")
    _save(fig, "07_zscore_credit_amount.png")
    return desc, rep
