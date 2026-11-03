"""Maximum-likelihood logistic regression with statsmodels -> odds ratios."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import plotstyle as ps
from .config import FIG_DIR, RES_DIR
from .preprocessing import selected_feature_names, transform_selected


def odds_ratio_table(pipe, X_tr, y_tr):
    names = selected_feature_names(pipe)
    Xt = pd.DataFrame(transform_selected(pipe, X_tr), columns=names)
    res = sm.Logit(y_tr.values, sm.add_constant(Xt)).fit(disp=0, maxiter=200)
    ci = res.conf_int()
    tab = pd.DataFrame({
        "coef": res.params, "std_err": res.bse, "z": res.tvalues, "p_value": res.pvalues,
        "odds_ratio": np.exp(res.params), "OR_CI_low": np.exp(ci[0]), "OR_CI_high": np.exp(ci[1]),
    }).round(4)
    tab.index.name = "feature"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RES_DIR / "odds_ratios.csv")
    (RES_DIR / "statsmodels_summary.txt").write_text(str(res.summary()))
    return tab, res


def plot_odds_ratios(tab, fname="13_odds_ratios.png"):
    t = tab.drop(index="const").sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [ps.BAD if v > 1 else ps.GOOD for v in t["odds_ratio"]]
    err = [t["odds_ratio"] - t["OR_CI_low"], t["OR_CI_high"] - t["odds_ratio"]]
    ax.barh(t.index, t["odds_ratio"] - 1, left=1, color=colors, xerr=err,
            error_kw={"ecolor": ps.TEXT, "capsize": 3})
    ax.axvline(1, color=ps.TEXT, lw=1)
    ax.set_xlabel("Odds ratio (>1 raises default risk, <1 lowers it)")
    ax.set_title("Logistic regression odds ratios with 95% CI (statsmodels, MLE)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
