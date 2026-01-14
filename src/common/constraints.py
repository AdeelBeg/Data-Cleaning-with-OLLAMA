from __future__ import annotations
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from .schema import RetailSchema

@dataclass
class ConstraintResult:
    name: str
    passed: int
    failed: int
    fail_rate_pct: float

def _count_fail(mask: pd.Series) -> tuple[int,int,float]:
    failed = int(mask.sum())
    total = int(mask.shape[0])
    passed = total - failed
    rate = (failed / total * 100) if total else 0.0
    return passed, failed, rate

def check_transaction_id(df: pd.DataFrame, s: RetailSchema) -> ConstraintResult:
    col = s.transaction_id
    missing = df[col].isna()
    bad_pattern = ~df[col].astype(str).str.match(r"^TXN_\d+$", na=False)
    dup = df[col].duplicated(keep=False) & df[col].notna()
    fail = missing | bad_pattern | dup
    passed, failed, rate = _count_fail(fail)
    return ConstraintResult("transaction_id_nonnull_pattern_unique", passed, failed, round(rate, 2))

def check_customer_id(df: pd.DataFrame, s: RetailSchema) -> ConstraintResult:
    col = s.customer_id
    missing = df[col].isna()
    bad_pattern = ~df[col].astype(str).str.match(r"^CUST_\d+$", na=False)
    fail = missing | bad_pattern
    passed, failed, rate = _count_fail(fail)
    return ConstraintResult("customer_id_nonnull_pattern", passed, failed, round(rate, 2))

def check_numeric_positive(df: pd.DataFrame, col: str, gt0: bool = True) -> ConstraintResult:
    x = pd.to_numeric(df[col], errors="coerce")
    missing_or_bad = x.isna()
    if gt0:
        invalid = ~(x > 0)
    else:
        invalid = ~(x >= 0)
    fail = missing_or_bad | invalid
    passed, failed, rate = _count_fail(fail)
    return ConstraintResult(f"{col}_numeric_{'gt0' if gt0 else 'ge0'}", passed, failed, round(rate, 2))

def check_quantity_int_ge1(df: pd.DataFrame, s: RetailSchema) -> ConstraintResult:
    col = s.quantity
    x = pd.to_numeric(df[col], errors="coerce")
    missing_or_bad = x.isna()
    int_like = (x.dropna() % 1 == 0)
    valid_int = pd.Series(False, index=df.index)
    valid_int.loc[x.dropna().index] = int_like.values
    ge1 = x >= 1
    fail = missing_or_bad | (~valid_int) | (~ge1)
    passed, failed, rate = _count_fail(fail)
    return ConstraintResult("quantity_int_ge1", passed, failed, round(rate, 2))

def check_transaction_date_parseable(df: pd.DataFrame, s: RetailSchema) -> ConstraintResult:
    col = s.transaction_date
    dt = pd.to_datetime(df[col], errors="coerce")
    fail = dt.isna()
    passed, failed, rate = _count_fail(fail)
    return ConstraintResult("transaction_date_parseable", passed, failed, round(rate, 2))

def check_arithmetic_total(df: pd.DataFrame, s: RetailSchema, tol: float = 1e-6) -> ConstraintResult:
    price = pd.to_numeric(df[s.price_per_unit], errors="coerce")
    qty = pd.to_numeric(df[s.quantity], errors="coerce")
    total = pd.to_numeric(df[s.total_spent], errors="coerce")
    expected = price * qty
    # Fail if any of the required numeric values are missing, or if mismatch beyond tolerance
    fail = price.isna() | qty.isna() | total.isna() | ((expected - total).abs() > tol)
    passed, failed, rate = _count_fail(fail)
    return ConstraintResult("arithmetic_total_price_times_qty", passed, failed, round(rate, 2))

def run_all_constraints(df: pd.DataFrame, s: RetailSchema) -> list[ConstraintResult]:
    results = [
        check_transaction_id(df, s),
        check_customer_id(df, s),
        check_numeric_positive(df, s.price_per_unit, gt0=True),
        check_quantity_int_ge1(df, s),
        check_numeric_positive(df, s.total_spent, gt0=False),
        check_transaction_date_parseable(df, s),
        check_arithmetic_total(df, s),
    ]
    return results
