from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..common.schema import RetailSchema
from ..traditional_cleaning.pipeline import clean_traditional
from .normalizer import genai_normalize_categoricals

def clean_genai(df: pd.DataFrame, s: RetailSchema, model: str, guardrails: bool, out_dir: str) -> pd.DataFrame:
    # Start from the same deterministic baseline transformations for fairness.
    df2 = clean_traditional(df, s)

    # GenAI: normalize category-like fields with Structured Outputs
    df3, _meta = genai_normalize_categoricals(df2, s, model=model, guardrails=guardrails, out_dir=out_dir)
    return df3
