# Loan Default Prediction using Logistic Regression and Discriminant Analysis

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



