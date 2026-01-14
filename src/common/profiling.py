from __future__ import annotations
import pandas as pd
import numpy as np
from .schema import RetailSchema

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2)
    }).sort_values("missing_pct", ascending=False)

def unique_counts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"unique_count": {c: df[c].nunique(dropna=True) for c in cols}})

def value_examples(df: pd.DataFrame, col: str, n: int = 10) -> list[str]:
    s = df[col].dropna().astype(str)
    return s.value_counts().head(n).index.tolist()

def infer_date_range(df: pd.DataFrame, col: str) -> tuple[str | None, str | None]:
    try:
        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.notna().sum() == 0:
            return None, None
        return str(dt.min().date()), str(dt.max().date())
    except Exception:
        return None, None
