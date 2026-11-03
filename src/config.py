"""Central configuration: paths, constants and column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "german_credit.csv"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K = 10
CV_FOLDS = 5

# German Credit cost matrix (from the original UCI documentation):
# approving a bad customer costs 5x more than rejecting a good one.
COST_FN = 5   # predicted good (0) but actually default (1)
COST_FP = 1   # predicted default (1) but actually good (0)

TARGET = "Risk"  # 1 = default / bad risk, 0 = good risk

NUM_COLS = ["duration_months", "credit_amount", "installment_rate", "residence_history",
            "age", "existing_credits", "dependents", "monthly_burden"]
# Only continuous variables are IQR-capped (small ordinal counts like `dependents`
# have IQR = 0, which would collapse them to a constant).
CAP_COLS = ["duration_months", "credit_amount", "age", "monthly_burden"]
CAT_COLS = ["checking_balance", "credit_history", "purpose", "savings_balance",
            "employment_length", "personal_status", "other_debtors", "property",
            "installment_plan", "housing", "telephone", "foreign_worker", "job", "age_group"]
BASE_CAT_COLS = [c for c in CAT_COLS if c != "age_group"]
AGE_BINS = [0, 25, 35, 50, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50+"]

# Attributes that raise fairness concerns in lending (used by --drop-sensitive)
SENSITIVE_COLS = ["personal_status", "foreign_worker", "age", "age_group"]

# Human-friendly labels for the app
LABELS = {
    "checking_balance": "Checking account balance",
    "duration_months": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings_balance": "Savings balance",
    "employment_length": "Employment length",
    "installment_rate": "Installment rate (% of income, 1-4)",
    "personal_status": "Personal status / sex",
    "other_debtors": "Other debtors / guarantors",
    "residence_history": "Years at present residence (1-4)",
    "property": "Property",
    "age": "Age (years)",
    "installment_plan": "Other installment plans",
    "housing": "Housing",
    "existing_credits": "Existing credits at this bank",
    "dependents": "Number of dependents",
    "telephone": "Telephone registered",
    "foreign_worker": "Foreign worker",
    "job": "Job type",
}
