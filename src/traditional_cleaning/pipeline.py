from __future__ import annotations
import pandas as pd
from ..common.schema import RetailSchema
from .steps import (
    standardize_strings, parse_booleans, cast_numerics, parse_dates,
    infer_price_or_total_when_possible, normalize_categoricals_simple,
)

def clean_traditional(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    df2 = standardize_strings(df, s)
    df2 = parse_booleans(df2, s)
    df2 = cast_numerics(df2, s)
    df2 = parse_dates(df2, s)
    df2 = normalize_categoricals_simple(df2, s)
    df2 = infer_price_or_total_when_possible(df2, s)
    return df2
