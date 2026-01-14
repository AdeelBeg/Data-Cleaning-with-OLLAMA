# audit_log.py
import csv
from pathlib import Path
from typing import List, Dict, Any

from config import RESULTS_DIR

AUDIT_LOG_PATH = RESULTS_DIR / "audit_log.csv"

AUDIT_FIELDS = [
    "method",      # traditional / genai
    "row_index",   # index in original dataframe
    "column",      # column that changed
    "old_value",   # value before cleaning
    "new_value",   # value after cleaning
    "note",        # e.g. median_imputation, mode_imputation, llm_correction
]


def init_audit_log() -> None:
    """
    Create the audit log file with header if it does not exist.
    Safe to call multiple times.
    """
    if not AUDIT_LOG_PATH.exists():
        with AUDIT_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            writer.writeheader()


def write_audit_entries(entries: List[Dict[str, Any]]) -> None:
    """
    Append a list of audit entries (dicts) to the audit log CSV.
    """
    if not entries:
        return

    with AUDIT_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        for entry in entries:
            # Ensure all required fields are present
            row = {field: entry.get(field, "") for field in AUDIT_FIELDS}
            writer.writerow(row)
