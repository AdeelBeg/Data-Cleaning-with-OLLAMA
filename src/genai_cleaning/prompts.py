from __future__ import annotations
from typing import Sequence

SYSTEM_NORMALIZE = """You are assisting with data cleaning for a retail transactions dataset.
Your task: normalize categorical values into a consistent canonical form.
Rules:
- Do NOT invent new categories; map each input to an existing canonical option whenever possible.
- Preserve meaning (e.g., 'credit card' and 'card' -> 'Credit Card').
- If an input is ambiguous, return the best guess but mark confidence low.
Return structured output only."""

def build_normalize_user_prompt(
    field_name: str,
    canonical_options: Sequence[str],
    observed_values: Sequence[str],
) -> str:
    canon = ", ".join([repr(x) for x in canonical_options])
    obs = ", ".join([repr(x) for x in observed_values])
    return f"""Normalize the field: {field_name}

Canonical options (choose one of these whenever possible):
{canon}

Observed raw values to normalize:
{obs}

Return a mapping for every observed raw value."""
