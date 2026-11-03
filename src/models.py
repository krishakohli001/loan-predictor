"""The nine classifiers: 3 required + 6 additional predictive models."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

CORE = ["Logistic Regression", "LDA", "QDA"]


def get_models():
    rs = RANDOM_STATE
    return {
        # --- required models ---
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=rs),
        "LDA": LinearDiscriminantAnalysis(),
        # reg_param stabilises QDA on one-hot (near-collinear) features
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.3),
        # --- additional predictive models ---
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "SVM (RBF)": SVC(probability=True, random_state=rs),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=rs),
        "Naive Bayes": GaussianNB(),
    }


# Small grids used with `python train.py --tune` (searched on the training set only)
TUNE_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "QDA": {"model__reg_param": [0.1, 0.3, 0.5, 0.7]},
    "Random Forest": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 5]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
                          "model__max_depth": [2, 3]},
    "SVM (RBF)": {"model__C": [0.5, 1, 3], "model__gamma": ["scale", 0.05]},
    "KNN": {"model__n_neighbors": [7, 15, 25]},
}


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
