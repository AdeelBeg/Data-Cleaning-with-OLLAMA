from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..common.schema import RetailSchema

ErrorType = Literal["missing_item", "corrupt_date", "break_total", "invalid_payment", "missing_price"]

@dataclass
class InjectionRecord:
    row_index: int
    field: str
    original: object
    corrupted: object
    error_type: ErrorType

def inject_errors(df: pd.DataFrame, s: RetailSchema, n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, list[InjectionRecord]]:
    """Inject controlled errors into a copy of df (assumed relatively clean subset).
    Returns corrupted_df, injection_records.
    """
    rng = np.random.default_rng(seed)
    df2 = df.copy()
    records: list[InjectionRecord] = []

    if df2.empty:
        return df2, records

    idx = rng.choice(df2.index.to_numpy(), size=min(n, len(df2)), replace=False)
    error_types: list[ErrorType] = ["missing_item", "corrupt_date", "break_total", "invalid_payment", "missing_price"]

    for i, row_i in enumerate(idx):
        et = error_types[i % len(error_types)]
        if et == "missing_item":
            field = s.item
            orig = df2.at[row_i, field]
            df2.at[row_i, field] = pd.NA
            records.append(InjectionRecord(int(row_i), field, orig, pd.NA, et))
        elif et == "corrupt_date":
            field = s.transaction_date
            orig = df2.at[row_i, field]
            df2.at[row_i, field] = "2024-99-99"
            records.append(InjectionRecord(int(row_i), field, orig, "2024-99-99", et))
        elif et == "break_total":
            field = s.total_spent
            orig = df2.at[row_i, field]
            # multiply by random factor
            df2.at[row_i, field] = float(orig) * 1.37 if pd.notna(orig) else 999999.0
            records.append(InjectionRecord(int(row_i), field, orig, df2.at[row_i, field], et))
        elif et == "invalid_payment":
            field = s.payment_method
            orig = df2.at[row_i, field]
            df2.at[row_i, field] = "PayBuddy"  # invalid on purpose
            records.append(InjectionRecord(int(row_i), field, orig, "PayBuddy", et))
        elif et == "missing_price":
            field = s.price_per_unit
            orig = df2.at[row_i, field]
            df2.at[row_i, field] = pd.NA
            records.append(InjectionRecord(int(row_i), field, orig, pd.NA, et))

    return df2, records

def score_repairs(
    cleaned_df: pd.DataFrame,
    original_df: pd.DataFrame,
    injections: list[InjectionRecord],
) -> dict:
    """Compute simple repair precision/recall on injected cells only."""
    if not injections:
        return {"repairs_total": 0, "repaired_correct": 0, "repaired_incorrect": 0, "unchanged": 0}

    repaired_correct = 0
    repaired_incorrect = 0
    unchanged = 0

    for rec in injections:
        ridx = rec.row_index
        field = rec.field
        # The "correct" value is the original
        correct = original_df.at[ridx, field] if ridx in original_df.index else rec.original
        got = cleaned_df.at[ridx, field] if ridx in cleaned_df.index else None

        if pd.isna(got) and pd.isna(rec.corrupted):
            # remained corrupted/missing
            unchanged += 1
        elif (pd.isna(got) and pd.isna(correct)) or (str(got) == str(correct)):
            repaired_correct += 1
        else:
            repaired_incorrect += 1

    total = len(injections)
    return {
        "repairs_total": total,
        "repaired_correct": repaired_correct,
        "repaired_incorrect": repaired_incorrect,
        "unchanged": unchanged,
        "repair_accuracy": round(repaired_correct / total, 4),
    }
