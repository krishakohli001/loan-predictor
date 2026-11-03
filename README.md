# 🏦 Loan Default Prediction using Logistic Regression, LDA, QDA and 6 more models

Predict whether a loan applicant will **default** using the German Credit data (1000 applicants), compare **9 classifiers**, and explore everything in an **interactive Streamlit web app**.

> **Problem.** Banks lose money when borrowers default, and turn away good customers when they are too cautious. This project estimates each applicant's probability of default and compares models on accuracy, recall, ROC-AUC and a *business cost* (a missed defaulter costs 5× a wrongly rejected good customer, per the original dataset documentation).

## Dataset

* **German Credit Data** (Statlog, Prof. Hans Hofmann, University of Hamburg) — 1000 loans, 20 attributes, 30 % bad risk. Original: UCI Machine Learning Repository, *Statlog (German Credit Data)*.
* `data/german_credit.csv` is a labelled copy of that dataset (public mirror: `stedy/Machine-Learning-with-R-datasets`). Its `default` column (1 = good, 2 = bad) is converted to **`Risk`** (1 = default / bad risk, 0 = good risk).
* Amounts are in Deutsche Marks, data from 1994 — a teaching dataset, not a modern scorecard.


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

**Link to the app-** https://loan-predictor-af9njm6lzwiete4mwktsfw.streamlit.app/#interactive-explorer
