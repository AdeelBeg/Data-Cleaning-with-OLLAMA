from __future__ import annotations
import pandas as pd
import numpy as np
from ..common.schema import RetailSchema

def standardize_strings(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    df = df.copy()
    for col in [s.category, s.item, s.payment_method, s.location, s.discount_applied, s.transaction_id, s.customer_id]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None", "NULL"]), col] = pd.NA
    return df

def parse_booleans(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    df = df.copy()
    col = s.discount_applied
    if col not in df.columns:
        return df
    mapping = {
        "TRUE": True, "True": True, "true": True, "1": True, "YES": True, "Yes": True, "Y": True,
        "FALSE": False, "False": False, "false": False, "0": False, "NO": False, "No": False, "N": False
    }
    # keep NA as NA
    df[col] = df[col].map(mapping).astype("boolean")
    return df

def cast_numerics(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    df = df.copy()
    for col in [s.price_per_unit, s.quantity, s.total_spent]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # quantity to int if int-like
    if s.quantity in df.columns:
        q = df[s.quantity]
        int_like = q.dropna().apply(lambda x: float(x).is_integer())
        df.loc[int_like.index, s.quantity] = q.loc[int_like.index].astype("Int64")
    return df

def parse_dates(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    df = df.copy()
    col = s.transaction_date
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def infer_price_or_total_when_possible(df: pd.DataFrame, s: RetailSchema, tol: float = 1e-9) -> pd.DataFrame:
    """Infer missing Price Per Unit or Total Spent using arithmetic consistency.
    Only fills values if the inference is exact (within tolerance) and avoids overriding existing values.
    """
    df = df.copy()
    price = df[s.price_per_unit]
    qty = df[s.quantity]
    total = df[s.total_spent]

    # infer price when missing and total/qty available
    mask_price = price.isna() & qty.notna() & total.notna() & (qty != 0)
    inferred_price = (total / qty).where(mask_price)
    df.loc[mask_price, s.price_per_unit] = inferred_price

    # infer total when missing and price/qty available
    mask_total = total.isna() & price.notna() & qty.notna()
    inferred_total = (price * qty).where(mask_total)
    df.loc[mask_total, s.total_spent] = inferred_total

    return df

def normalize_categoricals_simple(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    """Simple deterministic normalization (title case + whitespace collapse).
    Used as a baseline; GenAI aims to do better for synonyms/variants.
    """
    df = df.copy()
    for col in [s.category, s.payment_method, s.location]:
        if col in df.columns:
            df[col] = df[col].astype("string")
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
            df[col] = df[col].str.title()
    return df
