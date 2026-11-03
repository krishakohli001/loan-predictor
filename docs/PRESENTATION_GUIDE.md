# Presentation & Viva Guide

## 30-second pitch
"Banks lose money on defaulters and lose business by rejecting good customers. I used the German Credit dataset (1000 loans) to predict default. I built a leak-free pipeline — cleaning, IQR outlier capping, feature engineering, ANOVA-F feature selection — and compared **nine models**: Logistic Regression, LDA and QDA, plus Random Forest, Gradient Boosting, SVM, KNN, Decision Tree and Naive Bayes. Random Forest gave the highest accuracy (74 % test, 76 % in repeated CV with all features) against a 70 % baseline. I also estimated logistic-regression coefficients by maximum likelihood to explain *which factors* drive default, and wrapped everything in an interactive app."

## Slide outline (≈ 12 slides, 10-12 min)
1. **Title + problem** — cost of defaults vs cost of rejecting good customers.
2. **Dataset** — 1000 loans, 20 attributes, 30 % default (class imbalance → baseline 70 %). Show `01_pie_risk_distribution.png`.
3. **Cleaning & outliers** — median/mode imputation (no missing values in this file, code handles it); IQR fences `Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`; 70 duration, 72 amount, 23 age, 45 burden outliers capped. Show `04_boxplots_outliers.png`.
4. **EDA statistics** — mean/median/mode/variance/std/z-scores (`descriptive_statistics.csv`). Credit amount is right-skewed (skew 1.95, mean 3271 > median 2320; 25 points with |z| > 3).
5. **Visualisations** — bar (default rate by checking balance), histogram, heat-map, scatter. Key insight: *no checking account* → lowest default rate; *negative balance* → highest; longer duration & larger amount → more default.
6. **Feature engineering & selection** — `monthly_burden`, `age_group`; `SelectKBest(f_classif)` keeps top 10.
7. **Models** — what each one is (see cheat-sheet below). Show pipeline diagram: cap → scale/one-hot → select → model.
8. **Maximum-likelihood coefficients / odds ratios** — `13_odds_ratios.png`.
9. **Evaluation metrics** — confusion matrix, precision, recall, F1, ROC-AUC (`12_confusion_matrices.png`, `11_roc_curves.png`).
10. **Comparison & winner** — `10_model_comparison.png`, repeated-CV chart `16/17`. **Winner: Random Forest.**
11. **Live demo** — app: load random applicant, change checking balance / duration, flip the model, move the threshold.
12. **Limitations, ethics, future work.**

## Live demo script (2 min)
1. Open the app → **Predict** tab. Click *Load random real applicant* → note P(default) and the "actually defaulted/repaid" message.
2. Set checking balance to `< 0 DM`, duration to 48, amount 15000 → probability jumps; point at the "Why?" chart.
3. Drag the **threshold** to 0.17: "a bank that loses 5× more on a bad loan should reject earlier" → more applicants flagged.
4. Switch model to QDA/Random Forest → show that models disagree on borderline cases (bar chart of all 9 votes).
5. **Model comparison** tab → leaderboard, then repeated-CV tables.

## Model cheat-sheet
* **Logistic Regression** — models log-odds of default as a linear function; coefficients → odds ratios (`e^β`); estimated by maximum likelihood.
* **LDA** — assumes each class is Gaussian with the **same covariance**; gives a *linear* boundary. Close to logistic regression in practice.
* **QDA** — same but **each class has its own covariance** → *quadratic* boundary; more flexible, needs more data; we use `reg_param=0.3` because one-hot columns are near-collinear.
* **Random Forest** — many decorrelated decision trees, majority vote; handles non-linearity/interactions.
* **Gradient Boosting** — trees added sequentially to fix previous errors.
* **SVM (RBF)** — maximum-margin boundary in a kernel space. **KNN** — votes of 15 nearest applicants. **Decision Tree** — interpretable if-else rules. **Naive Bayes** — Bayes rule with independence assumption.

## Metrics in one line each
* **Accuracy** — share of correct predictions (misleading with 70/30 classes).
* **Precision** — of those flagged as defaulters, how many truly default. **Recall** — of all true defaulters, how many we caught.
* **F1** — harmonic mean of precision and recall. **ROC-AUC** — probability a random defaulter gets a higher score than a random good customer (threshold-free).
* **Business cost** — `5·FN + 1·FP` (original cost matrix).

## Honest talking points (examiners like these)
* The winner is only ~4 points above the always-approve baseline, and the top models are within the noise (CV std ≈ 2 points). Say so — it shows understanding.
* A single 200-loan test split can mislead: Gradient Boosting scored 80.5 % on one split with all features but 75.6 % in repeated CV. That is why the repeated-CV ranking is the more trustworthy one.
* Feature selection with a univariate ANOVA test reduced accuracy for flexible models (~2-3 points) because it ignores interactions; kept to follow the brief and shown as an experiment against "all features".
* Default is the positive class. Recall for most models is 35-50 %: the models miss many defaulters at the 0.5 threshold. Lowering the threshold trades precision for recall.
* One odds ratio looks counter-intuitive (`credit_history = fully repaid` OR 2.5 vs the reference "critical"). Explain it as a known quirk of this dataset's category definitions / small group size (wide CI 1.1-5.7), not a causal claim.

## Likely viva questions
**Why split before capping outliers?** Fences are computed on training data only; otherwise test information leaks into training.
**Why one-hot with `drop='first'`?** Avoids perfect multicollinearity (dummy trap) so statsmodels can estimate coefficients; the dropped level is the reference.
**What does OR = 1.48 for duration mean?** Each +1 SD (≈ 12 months) of duration multiplies the odds of default by 1.48, other features fixed.
**Why do LDA and logistic regression give similar results?** Both are linear; LDA assumes Gaussian classes with equal covariance, logistic regression makes no distributional assumption, but with mostly dummy features they land close together.
**Why is QDA's recall low?** Quadratic boundaries with many binary features overfit class-specific covariances; regularisation helps but recall stays lower.
**What is class imbalance and how did you handle it?** 30 % defaults; we used stratified splits and judged models with recall/AUC/cost, not accuracy alone. Next steps: class weights, SMOTE, threshold tuning.
**Is the model fair?** Sex/marital status, foreign-worker and age are sensitive. `--drop-sensitive` measures the cost of excluding them; real systems need fairness audits and legal review.
**How would you improve it?** Hyper-parameter tuning (`--tune`), class weighting, threshold optimisation by cost, calibration, bigger/modern data (e.g. Lending Club), SHAP explanations.

## Where each requirement lives
Cleaning → `src/data.py` · Outliers → `src/preprocessing.py` · EDA → `src/eda.py` · Charts → `reports/figures` · Feature engineering → `src/data.py` · SelectKBest → `src/preprocessing.py` · Models → `src/models.py` · statsmodels odds ratios → `src/stats_model.py` · Evaluation → `src/evaluate.py` · App → `app.py`.
