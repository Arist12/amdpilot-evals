#!/usr/bin/env python3
"""Extract task-plane validation specs from enriched PR registry rows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.validation_tools import build_validation_spec  # noqa: E402


def load_batch(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        prs = data.get("prs")
        if isinstance(prs, list):
            return prs
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported batch format in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract validation specs from enriched PR metadata",
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_batch(args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    enriched_rows: list[dict[str, Any]] = []
    tier_counts = {1: 0, 2: 0, 3: 0}

    for row in rows:
        spec = build_validation_spec(row)
        enriched_row = {**row, "validation_spec": spec}
        enriched_rows.append(enriched_row)
        tier_counts[spec["tier"]] = tier_counts.get(spec["tier"], 0) + 1

    payload = {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_rows": len(rows),
            "tier_1_has_tests": tier_counts[1],
            "tier_2_key_files_only": tier_counts[2],
            "tier_3_minimal": tier_counts[3],
        },
        "prs": enriched_rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
