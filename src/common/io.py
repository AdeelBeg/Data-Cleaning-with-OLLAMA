from __future__ import annotations
import pandas as pd
from pathlib import Path

def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv("/Users/adeel28/Documents/Adeel /Dessert'n/coding/olama/dissertation_cleaning_codebase_ollama_qwen2.5_7b/retail_store_sales.csv")

def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
