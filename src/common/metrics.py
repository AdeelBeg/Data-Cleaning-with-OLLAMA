from __future__ import annotations
import pandas as pd
from dataclasses import asdict
from .schema import RetailSchema
from .constraints import run_all_constraints

def quality_report(df: pd.DataFrame, s: RetailSchema) -> dict:
    constraints = run_all_constraints(df, s)
    constraint_dicts = [asdict(c) for c in constraints]
    # simple aggregate score: 100 - avg fail rate
    avg_fail = sum(c["fail_rate_pct"] for c in constraint_dicts) / max(len(constraint_dicts), 1)
    score = round(100.0 - avg_fail, 2)
    return {
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "constraint_results": constraint_dicts,
        "quality_score": score,
    }
