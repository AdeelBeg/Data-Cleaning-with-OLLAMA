from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_run_metadata(out_dir: str | Path, metadata: dict) -> None:
    metadata = dict(metadata)
    metadata["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
    write_json(metadata, Path(out_dir) / "run_metadata.json")
