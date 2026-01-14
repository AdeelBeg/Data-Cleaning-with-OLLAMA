from __future__ import annotations
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(df[x].astype(str), df[y].astype(float))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
