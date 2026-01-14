from __future__ import annotations
import pandas as pd
import numpy as np
from ..common.schema import RetailSchema
from ..common.metrics import quality_report

def build_metrics_row(df: pd.DataFrame, s: RetailSchema, label: str) -> dict:
    qr = quality_report(df, s)
    row = {"approach": label, "quality_score": qr["quality_score"], "rows": qr["row_count"]}
    for c in qr["constraint_results"]:
        row[f"fail_rate_{c['name']}"] = c["fail_rate_pct"]
    return row

def stability_against_reference(reference: pd.DataFrame, candidate: pd.DataFrame, columns: list[str]) -> float:
    """Fraction of identical cells over selected columns."""
    ref = reference[columns].astype(str)
    cand = candidate[columns].astype(str)
    eq = (ref.values == cand.values)
    return float(eq.mean())

def compute_reliability_stats(outputs: list[pd.DataFrame], cols: list[str]) -> dict:
    if len(outputs) <= 1:
        return {"runs": len(outputs), "mean_stability": 1.0, "min_stability": 1.0, "max_stability": 1.0}
    ref = outputs[0]
    stabs = [stability_against_reference(ref, o, cols) for o in outputs[1:]]
    return {
        "runs": len(outputs),
        "mean_stability": float(np.mean(stabs)),
        "min_stability": float(np.min(stabs)),
        "max_stability": float(np.max(stabs)),
    }
