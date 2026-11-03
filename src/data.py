"""Loading, cleaning and feature engineering."""
import pandas as pd

from .config import AGE_BINS, AGE_LABELS, BASE_CAT_COLS, DATA_PATH, TARGET

RENAME = {"months_loan_duration": "duration_months", "amount": "credit_amount"}
FIX_VALUES = {"mangement self-employed": "management / self-employed"}


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median for numeric columns, mode for categorical columns."""
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw file -> clean dataframe with binary target `Risk` (1 = default)."""
    df = raw.rename(columns=RENAME)
    df = clean_missing(df)
    for c in BASE_CAT_COLS:
        df[c] = df[c].astype(str).str.strip().replace(FIX_VALUES)
    # In the source file `default` is coded 1 = good, 2 = bad
    df[TARGET] = (df["default"] == 2).astype(int)
    return df.drop(columns=["default"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: monthly repayment burden and age groups."""
    df = df.copy()
    df["monthly_burden"] = df["credit_amount"] / df["duration_months"]
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    return df
