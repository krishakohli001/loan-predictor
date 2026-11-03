"""Interactive Loan Default Predictor  -  run with:  streamlit run app.py"""
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import plotstyle as ps
from src.config import COST_FN, COST_FP, FIG_DIR, LABELS, MODEL_DIR, RES_DIR, TARGET
from src.data import add_features, load_raw, prepare
from src.models import CORE, slug
from src.preprocessing import selected_feature_names, transform_selected

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.card {background: linear-gradient(145deg,#16335f,#0f2447); border-radius: 18px; padding: 18px 22px;
       box-shadow: 7px 7px 16px #060f20, -4px -4px 12px #18386e; margin-bottom: 14px;}
.big {font-size: 2.3rem; font-weight: 800; line-height: 1.1;}
.small {opacity: .75; font-size: .85rem;}
.bar {height: 16px; border-radius: 10px; background: linear-gradient(90deg,#4ade80,#fbbf24,#f87171); position: relative;}
.marker {position: absolute; top: -5px; width: 6px; height: 26px; background: #fff; border-radius: 3px;
         box-shadow: 0 0 8px #000;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>""", unsafe_allow_html=True)

ORDER = {
    "checking_balance": ["< 0 DM", "1 - 200 DM", "> 200 DM", "unknown"],
    "savings_balance": ["< 100 DM", "101 - 500 DM", "501 - 1000 DM", "> 1000 DM", "unknown"],
    "employment_length": ["unemployed", "0 - 1 yrs", "1 - 4 yrs", "4 - 7 yrs", "> 7 yrs"],
}
NUM_RANGES = {  # key: (min, max, step)
    "duration_months": (4, 72, 1), "credit_amount": (250, 20000, 50), "installment_rate": (1, 4, 1),
    "residence_history": (1, 4, 1), "age": (19, 75, 1), "existing_credits": (1, 4, 1), "dependents": (1, 2, 1),
}
INPUT_ORDER = list(LABELS.keys())


@st.cache_data
def get_data():
    return add_features(prepare(load_raw()))


@st.cache_resource(show_spinner="Loading models (first run trains them, ~1 min)...")
def load_models():
    try:
        meta = json.loads((RES_DIR / "summary.json").read_text())
        models = {n: joblib.load(MODEL_DIR / f"{slug(n)}.joblib") for n in meta["models"]}
        probe = add_features(prepare(load_raw())).drop(columns=[TARGET]).head(2)
        for m in models.values():
            m.predict_proba(probe)
    except Exception:  # missing files or scikit-learn version mismatch -> retrain
        import train
        train.run(verbose=False)
        meta = json.loads((RES_DIR / "summary.json").read_text())
        models = {n: joblib.load(MODEL_DIR / f"{slug(n)}.joblib") for n in meta["models"]}
    return models, meta


df = get_data()
models, meta = load_models()
cmp_top10 = pd.read_csv(RES_DIR / "model_comparison_top10.csv", index_col="Rank")
cmp_all = pd.read_csv(RES_DIR / "model_comparison_all_features.csv", index_col="Rank")


def options(col):
    vals = list(df[col].unique())
    return [v for v in ORDER[col] if v in vals] if col in ORDER else sorted(vals)


def defaults():
    d = {}
    for c in INPUT_ORDER:
        d[c] = int(df[c].median()) if c in NUM_RANGES else df[c].mode().iloc[0]
    return d


for k, v in defaults().items():
    st.session_state.setdefault(k, v)


def load_random():
    row = df.sample(1).iloc[0]
    for c in INPUT_ORDER:
        v = row[c]
        if c in NUM_RANGES:
            lo, hi, _ = NUM_RANGES[c]
            v = int(min(max(v, lo), hi))
        st.session_state[c] = v
    st.session_state["actual"] = "DEFAULTED" if row[TARGET] == 1 else "repaid / good risk"


def reset_form():
    for k, v in defaults().items():
        st.session_state[k] = v
    st.session_state.pop("actual", None)


# ------------------------------------------------------------------ header
st.title("🏦 Loan Default Predictor")
st.caption("German Credit data · 9 classifiers (Logistic Regression, LDA, QDA + 6 more) · "
           "top-10 ANOVA-F features · built with scikit-learn & statsmodels")

with st.sidebar:
    st.header("⚙️ Settings")
    best = meta["best_model"]
    names = list(models.keys())
    model_name = st.selectbox("Model used for the prediction", names, index=names.index(best),
                              help=f"Best test accuracy: {best}")
    threshold = st.slider("Decision threshold (P(default) ≥ → reject)", 0.05, 0.95, 0.50, 0.01,
                          help=f"Banks usually lower this: the original cost matrix (a bad loan costs "
                               f"{COST_FN}x a lost good one) implies ≈ {COST_FP/(COST_FP+COST_FN):.2f}.")
    st.markdown("---")
    st.metric("Best model (test accuracy)", f"{meta['best_accuracy']:.1%}", best)
    st.caption(f"Majority-class baseline: {meta['baseline_accuracy']:.0%}")

tab_pred, tab_cmp, tab_eda, tab_or, tab_batch, tab_about = st.tabs(
    ["🔮 Predict", "🏆 Model comparison", "📊 EDA", "📈 Odds ratios", "📁 Batch scoring", "ℹ️ About"])

# ------------------------------------------------------------------ predict
with tab_pred:
    c_btn1, c_btn2, c_act = st.columns([1, 1, 3])
    c_btn1.button("🎲 Load random real applicant", on_click=load_random)
    c_btn2.button("↩️ Reset", on_click=reset_form)
    if "actual" in st.session_state:
        c_act.info(f"This real applicant actually **{st.session_state['actual']}**.")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Applicant profile")
        cols = st.columns(2)
        for i, c in enumerate(INPUT_ORDER):
            with cols[i % 2]:
                if c in NUM_RANGES:
                    lo, hi, step = NUM_RANGES[c]
                    if c in ("credit_amount", "duration_months", "age"):
                        st.slider(LABELS[c], lo, hi, key=c, step=step)
                    else:
                        st.number_input(LABELS[c], lo, hi, key=c, step=step)
                else:
                    st.selectbox(LABELS[c], options(c), key=c)

    applicant = pd.DataFrame([{c: st.session_state[c] for c in INPUT_ORDER}])
    feats = add_features(applicant)
    probs = {n: float(m.predict_proba(feats)[0, 1]) for n, m in models.items()}
    p = probs[model_name]
    reject = p >= threshold

    with right:
        st.subheader("Result")
        colr = "#f87171" if reject else "#4ade80"
        verdict = "⚠️ HIGH RISK - likely to default" if reject else "✅ LOW RISK - likely to repay"
        st.markdown(f"""<div class="card"><div class="small">{model_name}</div>
            <div class="big" style="color:{colr}">{p:.1%}</div>
            <div class="small">estimated probability of default</div><br>
            <div class="bar"><div class="marker" style="left:{p*100:.1f}%"></div></div>
            <br><b style="color:{colr}">{verdict}</b>
            <div class="small">Recommendation at threshold {threshold:.2f}: <b>{'REJECT / review' if reject else 'APPROVE'}</b>
            · monthly burden = {feats['monthly_burden'].iloc[0]:.0f} DM</div></div>""", unsafe_allow_html=True)

        votes = sum(v >= threshold for v in probs.values())
        st.markdown(f"**{votes} of {len(probs)} models** flag this applicant as high risk.")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        s = pd.Series(probs).sort_values()
        ax.barh(s.index, s.values * 100, color=[ps.BAD if v >= threshold else ps.GOOD for v in s.values])
        ax.axvline(threshold * 100, color=ps.WARN, ls="--")
        ax.set_xlabel("P(default) %")
        ax.set_title("Every model's opinion")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        lr = models["Logistic Regression"]
        xt = transform_selected(lr, feats)[0]
        contrib = pd.Series(xt * lr.named_steps["model"].coef_[0], index=selected_feature_names(lr))
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(top.index, top.values, color=[ps.BAD if v > 0 else ps.GOOD for v in top.values])
        ax.set_title("Why? Logistic-regression log-odds contributions")
        ax.set_xlabel("← lowers risk | raises risk →")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------ comparison
with tab_cmp:
    best_row = cmp_top10.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Highest accuracy", best_row["Model"])
    c2.metric("Test accuracy", f"{best_row['Accuracy']:.1%}")
    c3.metric("Baseline (always 'good')", f"{meta['baseline_accuracy']:.0%}")
    c4.metric("Best ROC-AUC", f"{cmp_top10['ROC-AUC'].max():.3f}", cmp_top10.loc[cmp_top10['ROC-AUC'].idxmax(), 'Model'])
    st.info("Accuracy alone is misleading here: 70% of applicants are good, so a model that approves everyone "
            "already scores 70%. Look at **recall** (defaulters caught), **ROC-AUC** and the **business cost** "
            f"(a missed defaulter costs {COST_FN}x a wrongly rejected good customer).")

    def show(tbl):
        cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Accuracy", "Business Cost"]
        t = tbl[cols]
        try:
            st.dataframe(t.style.format({c: "{:.3f}" for c in cols[1:7]})
                         .highlight_max(subset=cols[1:7], color="#1f6f4a")
                         .highlight_min(subset=["Business Cost"], color="#1f6f4a"))
        except Exception:
            st.dataframe(t)

    st.subheader("Top-10 selected features (project spec)")
    show(cmp_top10)
    st.image(str(FIG_DIR / "10_model_comparison.png"))
    a, b = st.columns(2)
    a.image(str(FIG_DIR / "11_roc_curves.png"))
    b.image(str(FIG_DIR / "12_confusion_matrices.png"))
    st.subheader("Extra experiment: all features, no feature selection")
    show(cmp_all)
    st.image(str(FIG_DIR / "14_model_comparison_all_features.png"))

    st.subheader("Most reliable ranking: repeated 5-fold CV on all 1000 loans")
    st.caption("One 200-loan test split is noisy (±3 points). Averaging 15 train/test rounds gives a fairer winner.")
    r1, r2 = st.columns(2)
    for col, f, ttl in [(r1, "repeated_cv_top10.csv", "Top-10 features"), (r2, "repeated_cv_all_features.csv", "All features")]:
        t = pd.read_csv(RES_DIR / f, index_col="Rank")
        col.markdown(f"**{ttl}**")
        col.dataframe(t.round(3))
    r1.image(str(FIG_DIR / "16_repeated_cv_accuracy.png"))
    r2.image(str(FIG_DIR / "17_repeated_cv_all_features.png"))

# ------------------------------------------------------------------ EDA
with tab_eda:
    st.subheader("Interactive explorer")
    feat = st.selectbox("Pick a feature", [c for c in df.columns if c != TARGET],
                        format_func=lambda c: LABELS.get(c, c))
    fig, ax = plt.subplots(figsize=(9, 4))
    if feat in NUM_RANGES or feat == "monthly_burden":
        for r, lab in [(0, "Good"), (1, "Default")]:
            ax.hist(df.loc[df[TARGET] == r, feat], bins=30, alpha=.65, color=ps.RISK_COLORS[r], label=lab, density=True)
        ax.legend()
        ax.set_title(f"{feat}: distribution by Risk")
    else:
        rate = (df.groupby(feat)[TARGET].mean() * 100).sort_values()
        ax.barh(rate.index, rate.values, color=ps.ACC)
        ax.axvline(df[TARGET].mean() * 100, color=ps.WARN, ls="--", label="overall")
        ax.legend()
        ax.set_title(f"Default rate (%) by {feat}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Descriptive statistics (mean, median, mode, variance, std, z-scores)")
    st.dataframe(pd.read_csv(RES_DIR / "descriptive_statistics.csv", index_col=0))
    st.subheader("IQR outliers capped at the fences")
    st.dataframe(pd.read_csv(RES_DIR / "outlier_report.csv"))
    gallery = [("01_pie_risk_distribution.png", "Pie chart"), ("02_bar_default_rate_by_category.png", "Bar charts"),
               ("03_histograms.png", "Histograms"), ("04_boxplots_outliers.png", "Box plots (outliers)"),
               ("05_heatmap_correlation.png", "Heatmap"), ("06_scatter_amount_vs_duration.png", "Scatter plot"),
               ("07_zscore_credit_amount.png", "z-scores")]
    for f, cap in gallery:
        st.image(str(FIG_DIR / f), caption=cap)

# ------------------------------------------------------------------ odds ratios
with tab_or:
    st.subheader("Maximum-likelihood logistic regression (statsmodels)")
    st.caption("Numeric variables are standardised, so their odds ratio is per +1 standard deviation; "
               "dummy variables compare against the dropped reference category.")
    st.image(str(FIG_DIR / "13_odds_ratios.png"))
    st.dataframe(pd.read_csv(RES_DIR / "odds_ratios.csv", index_col=0))
    with st.expander("Full statsmodels summary"):
        st.code((RES_DIR / "statsmodels_summary.txt").read_text())
    st.image(str(FIG_DIR / "15_rf_feature_importance.png"))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Score many applicants at once")
    template = df.drop(columns=[TARGET, "monthly_burden", "age_group"]).head(5)
    st.download_button("⬇️ Download CSV template", template.to_csv(index=False), "applicants_template.csv")
    up = st.file_uploader("Upload a CSV with the same columns", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            feats_b = add_features(new)
            pb = models[model_name].predict_proba(feats_b)[:, 1]
            out = new.assign(P_default=pb.round(4), Decision=np.where(pb >= threshold, "REJECT / review", "APPROVE"))
            st.dataframe(out)
            st.download_button("⬇️ Download predictions", out.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"Could not score the file: {e}")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
### Pipeline
1. **Cleaning** - missing numeric → median, categorical → mode (this file has none, the code handles it).
2. **Outliers** - IQR rule, capped at fences (fitted on training data only).
3. **EDA** - mean, median, mode, variance, std, z-scores.
4. **Visualisation** - bar, histogram, pie, heatmap, scatter.
5. **Feature engineering** - `monthly_burden = credit_amount / duration`, `age_group`.
6. **Feature selection** - `SelectKBest(f_classif)` → top 10.
7. **Models** - Logistic Regression, LDA, QDA + Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, Naive Bayes.
8. **Coefficients** - statsmodels MLE → odds ratios.
9. **Evaluation** - accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves, business cost.

### Ethics & limitations
- Sex/marital status, foreign-worker status and age are **sensitive attributes**; a real bank may not use them.
  Run `python train.py --drop-sensitive` to see the accuracy cost of removing them.
- Data is from 1994 Germany (1000 loans, amounts in Deutsche Marks) - a teaching dataset, not a production scorecard.
- Test set is only 200 loans, so differences of 1-2 points between models are within noise.
""")
