from __future__ import annotations
import pandas as pd
from ..common.schema import RetailSchema
from ..common.metrics import quality_report

def accept_if_quality_not_worse(before_df: pd.DataFrame, after_df: pd.DataFrame, s: RetailSchema) -> bool:
    b = quality_report(before_df, s)["quality_score"]
    a = quality_report(after_df, s)["quality_score"]
    return a >= b
