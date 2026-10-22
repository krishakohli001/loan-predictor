# Loan Default Prediction using Logistic Regression and Discriminant Analysis

Predictive Analytics course project, Manipal University Jaipur.

**Team:** <Student 1 name> (<Reg. No.>), <Student 2 name> (<Reg. No.>) | **Section:** D

**Live demo:** <your GitHub Pages link>

## Problem statement

Banks lose money when borrowers default, and turn away good customers when they are too cautious. This project predicts whether a loan applicant is likely to default, using demographic and financial attributes, and compares three classification models.

## Dataset

- **Name:** German Credit Data (1000 applicants, good/bad credit risk)
- **Source:** <paste the UCI or Kaggle link you used>
- **Target:** `Risk` (1 = default / bad risk, 0 = good risk)

## Methodology

1. **Data cleaning:** missing values filled with the median (numeric) or mode (categorical).
2. **Outlier handling:** IQR rule with box plots, outliers capped at the fences.
3. **EDA:** mean, median, mode, variance, standard deviation and z-scores.
4. **Visualization:** bar chart, histogram, pie chart, heatmap, scatter plot.
5. **Feature engineering:** monthly repayment burden (credit amount / duration), age groups.
6. **Feature selection:** `SelectKBest` (ANOVA F-test), top 10 features.
7. **Modeling:** Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA).
8. **Coefficient estimation:** maximum likelihood with `statsmodels`, interpreted as odds ratios.
9. **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curves.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| LDA | | | | | |
| QDA | | | | | |

**Best model:** <name> because <one-line reason>.

**Key findings:**
- <Which features had the largest odds ratios and what that means>
- <How LDA and QDA compared with Logistic Regression, and why>

## Web app

`index.html` is an interactive loan risk predictor with a dark blue, tactile interface. It lets you:

- adjust applicant details and see the default probability on a live dial
- set an approval cut-off and see an Approve or Decline decision
- view which inputs raised or lowered the risk
- explore a sensitivity chart and compare pinned scenarios
- turn off age in the score to see the fairness effect

The coefficients in the app come from the Logistic Regression model in the notebook.

## Ethical considerations

- Features such as age, gender or foreign-worker status can lead to biased lending decisions.
- Default and approval rates should be checked across demographic groups before any model is used.
- False negatives (missed defaults) cost the lender, while false positives deny credit to good applicants. The approval cut-off controls this trade-off.
- Logistic regression is explainable through its odds ratios, which makes decisions easier to audit.

## Repository structure

```
loan-default-prediction/
├── data/                  # German Credit CSV
├── notebooks/             # Jupyter notebook with full analysis
├── index.html             # Interactive web app
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/loan-default-prediction.git
cd loan-default-prediction
pip install -r requirements.txt
jupyter notebook notebooks/
```

Open `index.html` in a browser to use the web app.

## Tech stack

Python, pandas, NumPy, scikit-learn, statsmodels, SciPy, Matplotlib, Seaborn, HTML/CSS/JavaScript.

## References

- German Credit Data, UCI Machine Learning Repository
- Course material: Predictive Analytics (logistic regression, discriminant analysis, EDA)
