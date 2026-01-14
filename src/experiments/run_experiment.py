from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from ..common.schema import RetailSchema
from ..common.io import read_csv, write_csv
from ..common.metrics import quality_report
from ..common.logging_utils import write_json, write_run_metadata
from ..traditional_cleaning.pipeline import clean_traditional
from ..genai_cleaning.pipeline import clean_genai
from ..evaluation.compare import build_metrics_row, compute_reliability_stats
from ..evaluation.plots import save_barplot
from ..evaluation.error_injection import inject_errors, score_repairs

def _audit_log(before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    # Record only cells that changed
    changes = []
AUDIT_FIELDS = [
    "method",
    "row_index",
    "column",
    "old_value",
    "new_value",
    "note",
]

def _collect_audit_entries(
    before: pd.DataFrame,
    after: pd.DataFrame,
    method: str,
    note: str,
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    common_cols = [c for c in before.columns if c in after.columns]
    for idx in before.index.intersection(after.index):
        b = before.loc[idx, common_cols]
        a = after.loc[idx, common_cols]
        diff_cols = [c for c in common_cols if str(b[c]) != str(a[c])]
        for c in diff_cols:
            changes.append({"row_index": int(idx), "column": c, "before": str(b[c]), "after": str(a[c])})
            changes.append({
                "method": method,
                "row_index": int(idx),
                "column": c,
                "old_value": str(b[c]),
                "new_value": str(a[c]),
                "note": note,
            })
    return changes

def _write_audit_log(entries: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(changes).to_csv(out_path, index=False)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for entry in entries:
            row = {field: entry.get(field, "") for field in AUDIT_FIELDS}
            writer.writerow(row)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_dir", required=True)
    # For Ollama, this should match an installed local model, e.g. qwen2.5:7b
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--no_genai", action="store_true")
    p.add_argument("--no_guardrails", action="store_true")
    p.add_argument("--reliability_runs", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = RetailSchema()
    raw = read_csv(args.input_csv)

    # Baseline reports
    raw_report = quality_report(raw, s)
    write_json(raw_report, out_dir / "quality_raw.json")

    # Traditional cleaning
    trad = clean_traditional(raw, s)
    trad_report = quality_report(trad, s)
    write_json(trad_report, out_dir / "quality_traditional.json")
    write_csv(trad, out_dir / "cleaned_traditional.csv")

    metrics_rows = [build_metrics_row(raw, s, "raw"), build_metrics_row(trad, s, "traditional")]
    audit_entries = _collect_audit_entries(raw, trad, "traditional", "traditional_cleaning")

    # GenAI cleaning (optional)
    genai_outputs = []
    if not args.no_genai:
        for i in range(max(1, args.reliability_runs)):
            run_dir = out_dir / f"genai_run_{i+1}"
            run_dir.mkdir(parents=True, exist_ok=True)
            g = clean_genai(raw, s, model=args.model, guardrails=(not args.no_guardrails), out_dir=str(run_dir))
            genai_outputs.append(g)
            write_csv(g, run_dir / "cleaned_genai.csv")
            write_json(quality_report(g, s), run_dir / "quality_genai.json")

        # Use first GenAI output as main reported output
        genai_main = genai_outputs[0]
        write_csv(genai_main, out_dir / "cleaned_genai.csv")
        metrics_rows.append(build_metrics_row(genai_main, s, "genai" + ("" if args.no_guardrails else "_guardrails")))

        # Reliability stats
        rel_cols = [s.category, s.payment_method, s.location, s.item, s.price_per_unit, s.quantity, s.total_spent, s.transaction_date]
        rel_cols = [c for c in rel_cols if c in genai_main.columns]
        rel = compute_reliability_stats(genai_outputs, rel_cols)
        write_json(rel, out_dir / "genai_reliability.json")

        # Audit log for main GenAI output
        _audit_log(trad, genai_main, out_dir / "audit_log_traditional_vs_genai.csv")
        # Audit log entries for main GenAI output
        genai_label = "genai_guardrails" if not args.no_guardrails else "genai"
        audit_entries.extend(_collect_audit_entries(trad, genai_main, genai_label, "genai_cleaning"))

    # Error-injection benchmark (optional, but valuable for dissertation)
    # Build a high-confidence subset: rows that have non-null key fields
    trad_clean_subset = trad.dropna(subset=[s.transaction_id, s.customer_id, s.price_per_unit, s.quantity, s.total_spent, s.transaction_date])
    # Keep a stable sample
    subset = trad_clean_subset.sample(n=min(1000, len(trad_clean_subset)), random_state=42) if len(trad_clean_subset) else trad_clean_subset
    corrupted, injections = inject_errors(subset, s, n=min(200, len(subset)), seed=42)

    # Run both cleaners on injected data
    trad_injected = clean_traditional(corrupted, s)
    injected_scores = {"traditional": score_repairs(trad_injected, subset, injections)}

    if not args.no_genai:
        genai_injected = clean_genai(corrupted, s, model=args.model, guardrails=(not args.no_guardrails), out_dir=str(out_dir / "genai_injected"))
        injected_scores["genai_guardrails" if not args.no_guardrails else "genai"] = score_repairs(genai_injected, subset, injections)

    write_json(injected_scores, out_dir / "injected_error_repair_scores.json")

    # Save metrics summary + a simple plot
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    save_barplot(metrics_df, x="approach", y="quality_score", title="Quality score by approach", out_path=out_dir / "figures" / "quality_score.png")

    write_run_metadata(out_dir, {
        "input_csv": str(Path(args.input_csv).resolve()),
        "model": args.model,
        "genai_enabled": not args.no_genai,
        "guardrails": not args.no_guardrails,
        "reliability_runs": args.reliability_runs,
    })

    _write_audit_log(audit_entries, out_dir / "audit_log.csv")

if __name__ == "__main__":
    main()