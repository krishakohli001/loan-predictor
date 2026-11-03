# 🏦 Loan Default Prediction — Logistic Regression, LDA, QDA and 6 more models

Predict whether a loan applicant will **default** using the German Credit data (1000 applicants), compare **9 classifiers**, and explore everything in an **interactive Streamlit web app**.

> **Problem.** Banks lose money when borrowers default, and turn away good customers when they are too cautious. This project estimates each applicant's probability of default and compares models on accuracy, recall, ROC-AUC and a *business cost* (a missed defaulter costs 5× a wrongly rejected good customer, per the original dataset documentation).

## Quick start

```bash
git clone https://github.com/<your-username>/loan-default-prediction.git
cd loan-default-prediction
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py               # re-creates every figure, table and model (≈ 5 min)
streamlit run app.py          # opens the interactive app at http://localhost:8501
```

Optional flags: `python train.py --tune` (grid-search hyper-parameters, slow) and `python train.py --drop-sensitive` (remove sex/marital status, foreign-worker and age to measure the fairness trade-off).
The app also trains the models by itself on first launch if `models/` is missing or was built with a different scikit-learn version.

## Dataset

* **German Credit Data** (Statlog, Prof. Hans Hofmann, University of Hamburg) — 1000 loans, 20 attributes, 30 % bad risk. Original: UCI Machine Learning Repository, *Statlog (German Credit Data)*.
* `data/german_credit.csv` is a labelled copy of that dataset (public mirror: `stedy/Machine-Learning-with-R-datasets`). Its `default` column (1 = good, 2 = bad) is converted to **`Risk`** (1 = default / bad risk, 0 = good risk).
* Amounts are in Deutsche Marks, data from 1994 — a teaching dataset, not a modern scorecard.

## Methodology (matches the project brief)

| # | Step | Where |
|---|------|-------|
| 1 | Data cleaning — median (numeric) / mode (categorical) imputation | `src/data.py` |
| 2 | Outliers — IQR rule + box plots, capped at the fences (fitted on **train only**) | `src/preprocessing.py`, `src/eda.py` |
| 3 | EDA — mean, median, mode, variance, std, z-scores | `reports/results/descriptive_statistics.csv` |
| 4 | Visualisation — bar, histogram, pie, heat-map, scatter (+ box, z-score) | `reports/figures/01…07` |
| 5 | Feature engineering — `monthly_burden = credit_amount / duration`, `age_group` | `src/data.py` |
| 6 | Feature selection — `SelectKBest(f_classif)`, top 10 | `src/preprocessing.py` |
| 7 | Models — Logistic Regression, LDA, QDA **+ Random Forest, Gradient Boosting, SVM (RBF), KNN, Decision Tree, Naive Bayes** | `src/models.py` |
| 8 | Maximum-likelihood coefficients with `statsmodels` → odds ratios | `src/stats_model.py` |
| 9 | Evaluation — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, business cost | `src/evaluate.py` |

Everything (capping → scaling/one-hot → feature selection → model) lives inside one scikit-learn `Pipeline`, so **no information from the test set leaks into training**. Split: 80/20 stratified, `random_state=42`.

## Results

Majority-class baseline (approve everybody): **70 %** accuracy.

### Single 80/20 split — top-10 features (200 test loans)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Business cost ↓ |
|:--|--:|--:|--:|--:|--:|--:|
| **Random Forest** | **0.740** | 0.583 | 0.467 | 0.519 | 0.740 | 180 |
| Logistic Regression | 0.720 | 0.553 | 0.350 | 0.429 | 0.765 | 212 |
| LDA | 0.720 | 0.548 | 0.383 | 0.451 | 0.764 | 204 |
| QDA | 0.720 | 0.571 | 0.267 | 0.364 | 0.763 | 232 |
| Gradient Boosting | 0.715 | 0.531 | 0.433 | 0.477 | 0.739 | 193 |
| Naive Bayes | 0.710 | 0.518 | 0.483 | 0.500 | 0.733 | 182 |
| KNN | 0.705 | 0.514 | 0.317 | 0.392 | 0.712 | 223 |
| SVM (RBF) | 0.700 | 0.500 | 0.233 | 0.318 | 0.731 | 244 |
| Decision Tree | 0.685 | 0.474 | 0.450 | 0.462 | 0.710 | 195 |

### More reliable: repeated 5-fold CV on all 1000 loans (3 repeats)

| Top-10 features | Accuracy | | All features | Accuracy |
|:--|--:|--|:--|--:|
| SVM (RBF) | 0.737 | | **Random Forest** | **0.761** |
| Random Forest | 0.731 | | Gradient Boosting | 0.756 |
| LDA | 0.729 | | SVM (RBF) | 0.752 |
| Logistic Regression | 0.729 | | Logistic Regression | 0.748 |
| Gradient Boosting | 0.728 | | LDA | 0.746 |
| QDA | 0.728 | | QDA | 0.738 |

### Conclusions

* **Highest accuracy:** *Random Forest* — 74.0 % on the top-10-feature test split and 76.1 % in repeated CV with all features (the best score in the repeated-CV ranking).
* The top models are separated by only ~1 point, which is **inside the noise** (CV std ≈ 2 points; 200 test loans ≈ ±3 points). The honest reading: tree ensembles ≈ SVM ≈ logistic regression/LDA/QDA on this dataset.
* The best model is only ~4-6 points above the 70 % baseline. Recall (defaulters caught) is 35-50 % for most models, so **accuracy alone flatters them** — check recall, ROC-AUC and business cost.
* Restricting to 10 ANOVA-F features costs ~2-3 points of accuracy for the flexible models — univariate selection discards useful interactions.
* **Statistical (odds-ratio) view** — strongest risk drivers (statsmodels MLE): longer duration (OR 1.48 per +1 SD), negative checking balance (1.52), other installment plans (no plan: 0.52), no checking account (0.28). Reference categories are the alphabetically first level of each variable.

Figures: `reports/figures/` · tables: `reports/results/` · full statsmodels output: `reports/results/statsmodels_summary.txt`.

![comparison](reports/figures/10_model_comparison.png)
![roc](reports/figures/11_roc_curves.png)
![odds](reports/figures/13_odds_ratios.png)

## The app (dark-blue UI)

* **Predict** — fill in an applicant (or load a random real one), choose any of the 9 models and a decision threshold; see P(default), every model's vote and the logistic-regression explanation of *why*.
* **Model comparison** — leaderboards, ROC curves, confusion matrices, repeated-CV ranking.
* **EDA**, **Odds ratios**, **Batch scoring** (upload CSV → download predictions), **About**.

## Deploy

**GitHub:** create an empty repo, then
```bash
git init && git add . && git commit -m "Loan default prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/loan-default-prediction.git
git push -u origin main
```
**Free live website (Streamlit Community Cloud):** go to <https://share.streamlit.io> → *New app* → pick this repo, branch `main`, main file `app.py` → *Deploy*. Add the resulting URL to this README.

## Project structure

```
app.py                      Streamlit app
train.py                    full pipeline (EDA → models → reports → saved models)
src/                        config, data, preprocessing, models, eda, evaluate, stats_model, plotstyle
data/german_credit.csv      dataset
models/*.joblib             trained pipelines (one per model)
reports/figures|results     charts and tables
notebooks/                  step-by-step notebook
docs/PRESENTATION_GUIDE.md  slide outline, talking points, viva Q&A
.streamlit/config.toml      dark-blue theme
```

## Ethics & limitations

Sex/marital status, foreign-worker status and age are sensitive attributes that many jurisdictions do not allow in credit decisions; they are kept here because they are part of the benchmark. Run `python train.py --drop-sensitive` to quantify the accuracy cost of removing them. The data is old, small and from one country; do not use this for real lending decisions.
