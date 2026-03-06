#!/usr/bin/env python3
"""Enrich PR registry rows with replay metadata for amdpilot-evals.

This script is task-plane tooling. It takes lightweight PR rows and adds:

- canonical PR metadata from GitHub
- deterministic replay-base derivation
- optional ground-truth diff export
- optional git apply verification
- normalized test commands
- GPU-required classification

It does not materialize eval instances and does not run the orchestrator.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.registry_tools import (  # noqa: E402
    classify_gpu_required,
    derive_replay_base,
    fetch_pr_diff,
    gh_api_json,
    key_files_to_list,
    manifest_path,
    normalize_test_commands,
    pr_slug,
    read_json_source,
    verify_apply_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich PR registry entries for amdpilot-evals",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="JSON source path or URL. Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output enriched registry JSON path.",
    )
    parser.add_argument(
        "--diff-dir",
        type=Path,
        default=None,
        help="Directory to store ground-truth diff files.",
    )
    parser.add_argument(
        "--apply-check",
        action="store_true",
        help="Verify replay_base_sha via git apply --check.",
    )
    parser.add_argument(
        "--apply-timeout-seconds",
        type=int,
        default=180,
        help="Timeout for each git command during apply check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.diff_dir:
        args.diff_dir.mkdir(parents=True, exist_ok=True)

    raw_entries: list[dict[str, Any]] = []
    for source in args.source:
        raw_entries.extend(read_json_source(source))

    seen: set[tuple[str, int]] = set()
    enriched_rows: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(raw_entries):
        repo = str(row.get("repo", "")).strip()
        pr_number_raw = row.get("pr_number")
        if not repo or pr_number_raw is None:
            skipped += 1
            continue
        pr_number = int(pr_number_raw)

        key = (repo, pr_number)
        if key in seen:
            continue
        seen.add(key)

        pr_data = gh_api_json(f"repos/{repo}/pulls/{pr_number}")
        replay_base_sha, strategy, strategy_details = derive_replay_base(
            repo, pr_number, pr_data,
        )

        diff_path_text = None
        apply_ok = None
        apply_message = "skipped"
        if args.diff_dir:
            diff_text = fetch_pr_diff(repo, pr_number)
            diff_path = args.diff_dir / f"{pr_slug(repo)}__pr{pr_number}.diff"
            diff_path.write_text(diff_text, encoding="utf-8")
            diff_path_text = manifest_path(diff_path, PROJECT_ROOT)
            if args.apply_check:
                apply_ok, apply_message = verify_apply_check(
                    repo=repo,
                    replay_base_sha=replay_base_sha,
                    diff_text=diff_text,
                    timeout_seconds=args.apply_timeout_seconds,
                )

        test_commands = normalize_test_commands(
            row.get("test_commands_normalized", row.get("test_commands")),
        )
        key_files = key_files_to_list(row.get("key_files", []))
        if not key_files:
            key_files = [f.get("filename", "") for f in pr_data.get("files", []) if f.get("filename")]

        enriched_rows.append(
            {
                **row,
                "repo": repo,
                "pr_number": pr_number,
                "source_index": idx,
                "title": pr_data.get("title", row.get("title", "")),
                "url": pr_data.get("html_url", row.get("url", "")),
                "created_at": pr_data.get("created_at"),
                "merged_at": pr_data.get("merged_at"),
                "state": pr_data.get("state"),
                "labels": [label.get("name", "") for label in pr_data.get("labels", [])],
                "base_sha": pr_data.get("base", {}).get("sha"),
                "head_sha": pr_data.get("head", {}).get("sha"),
                "merge_commit_sha": pr_data.get("merge_commit_sha"),
                "files_changed": len(pr_data.get("files", [])),
                "key_files": key_files,
                "test_commands_normalized": test_commands,
                "gpu_required": classify_gpu_required(repo, test_commands),
                "replay_base_sha": replay_base_sha,
                "replay_base_strategy": strategy,
                "replay_base_details": strategy_details,
                "ground_truth_diff_path": diff_path_text,
                "apply_check_ok": apply_ok,
                "apply_check_message": apply_message,
            }
        )

    payload = {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_count": len(args.source),
            "input_rows": len(raw_entries),
            "skipped_rows": skipped,
            "unique_prs": len(enriched_rows),
            "apply_check_enabled": bool(args.apply_check),
            "diff_exported": bool(args.diff_dir),
        },
        "prs": enriched_rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "unique_prs": len(enriched_rows),
                "skipped_rows": skipped,
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
