from __future__ import annotations
import pandas as pd
from typing import Sequence
from ..common.schema import RetailSchema
from ..common.metrics import quality_report
from ..common.logging_utils import write_json
from .llm_client import LLMClient
from .prompts import SYSTEM_NORMALIZE, build_normalize_user_prompt
from .types import NormalizationResponse

def _top_canonical(series: pd.Series, k: int = 10) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return []
    return s.value_counts().head(k).index.tolist()

def propose_mapping_for_field(
    llm: LLMClient,
    model: str,
    field_name: str,
    canonical_options: Sequence[str],
    observed_values: Sequence[str],
) -> dict[str, str]:
    resp = llm.parse(
        model=model,
        system=SYSTEM_NORMALIZE,
        user=build_normalize_user_prompt(field_name, canonical_options, observed_values),
        schema_model=NormalizationResponse,
    )
    mapping = {m.raw: m.normalized for m in resp.mappings}
    return mapping

def apply_mapping(df: pd.DataFrame, col: str, mapping: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    df[col] = df[col].astype("string")
    df[col] = df[col].map(lambda x: mapping.get(str(x), x) if x is not pd.NA else pd.NA)
    return df

def genai_normalize_categoricals(
    df: pd.DataFrame,
    s: RetailSchema,
    model: str,
    guardrails: bool,
    out_dir: str,
    max_values_per_field: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Uses a local LLM (via Ollama) to propose canonical mappings for Category, Payment Method, Location.
    Guardrails: accept mappings only if they do not degrade the overall quality score.
    """
    llm = LLMClient()
    meta: dict = {"model": model, "fields": {}}
    df2 = df.copy()

    for col in [s.category, s.payment_method, s.location]:
        if col not in df2.columns:
            continue

        # Canonical options: most frequent values after simple trim/title (baseline normalization).
        baseline = df2[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.title()
        canonical = _top_canonical(baseline, k=10)

        # Observed: unique raw values (limited) to avoid large prompts
        observed = df2[col].dropna().astype(str).value_counts().head(max_values_per_field).index.tolist()
        if not observed or not canonical:
            continue

        before = quality_report(df2, s)["quality_score"]
        mapping = propose_mapping_for_field(llm, model, col, canonical, observed)
        candidate = apply_mapping(df2, col, mapping)
        after = quality_report(candidate, s)["quality_score"]

        accepted = True
        if guardrails and (after + 1e-9) < before:
            accepted = False

        meta["fields"][col] = {
            "canonical_options": canonical,
            "observed_values_count": len(observed),
            "quality_before": before,
            "quality_after": after,
            "accepted": accepted,
        }
        if accepted:
            df2 = candidate

    write_json(meta, f"{out_dir}/genai_normalization_meta.json")
    return df2, meta
