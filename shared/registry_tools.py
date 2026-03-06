#!/usr/bin/env python3
"""Registry helpers for PR-derived eval tooling."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

GPU_HEAVY_REPOS = {
    "ROCm/aiter",
    "ROCm/composable_kernel",
    "ROCm/HIP",
    "ROCm/HIPIFY",
    "ROCm/rocm-libraries",
    "Dao-AILab/flash-attention",
    "triton-lang/triton",
    "pytorch/pytorch",
    "sgl-project/sglang",
    "vllm-project/vllm",
}

GPU_HINTS = ("rocm", "hip", "cuda", "gpu", "gfx", "mi300", "mi350", "mi355", "rocblas")

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def pr_slug(repo: str) -> str:
    """Filesystem-safe slug for a repo name."""
    return _SLUG_RE.sub("_", repo)


def normalize_test_commands(value: Any) -> list[str]:
    """Normalize test commands from a string or list into ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def key_files_to_list(value: Any) -> list[str]:
    """Coerce key_files from a comma-separated string or list to ``list[str]``."""
    if isinstance(value, list):
        return [str(f).strip() for f in value if str(f).strip()]
    if isinstance(value, str):
        return [f.strip() for f in value.split(",") if f.strip()]
    return []


def parse_pr_ref(pr_ref: str) -> tuple[str, int]:
    """Parse ``owner/repo/NUMBER`` or full GitHub PR URL into ``(repo, number)``."""
    match = re.match(
        r"(?:https?://github\.com/)?([^/]+/[^/]+?)(?:/pull)?/(\d+)/?$",
        pr_ref,
    )
    if not match:
        raise ValueError(f"Cannot parse PR reference: {pr_ref}")
    return match.group(1), int(match.group(2))


def read_json_source(source: str) -> list[dict[str, Any]]:
    """Read a registry source from a local file or URL."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "prs" in data and isinstance(data["prs"], list):
            return data["prs"]
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure from {source}")


def gh_api(path: str, *, accept: str | None = None) -> str:
    """Call the GitHub API via ``gh api``."""
    cmd = ["gh", "api"]
    if accept:
        cmd.extend(["-H", f"Accept: {accept}"])
    cmd.append(path)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api failed for {path}: {proc.stderr.strip()}")
    return proc.stdout


def gh_api_json(path: str) -> Any:
    return json.loads(gh_api(path))


def classify_gpu_required(repo: str, test_commands: list[str]) -> bool:
    """Infer whether a task requires GPU access."""
    if repo in GPU_HEAVY_REPOS:
        return True
    joined = " ".join(test_commands).lower()
    return any(hint in joined for hint in GPU_HINTS)


def derive_replay_base(
    repo: str,
    pr_number: int,
    pr_data: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Derive the best replay base SHA for a PR."""
    base_sha = pr_data.get("base", {}).get("sha", "")
    head_sha = pr_data.get("head", {}).get("sha", "")
    merge_sha = pr_data.get("merge_commit_sha", "")

    details: dict[str, Any] = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "merge_commit_sha": merge_sha,
        "strategy": "base_sha_fallback",
        "notes": [],
    }

    if merge_sha:
        try:
            merge_commit = gh_api_json(f"repos/{repo}/commits/{merge_sha}")
            parents = [
                p.get("sha", "")
                for p in merge_commit.get("parents", [])
                if p.get("sha")
            ]
            details["merge_commit_parents"] = parents
            if len(parents) >= 2:
                details["strategy"] = "merge_commit_parent"
                return parents[0], details["strategy"], details
            if len(parents) == 1:
                details["strategy"] = "squash_or_rebase_parent"
                return parents[0], details["strategy"], details
            details["notes"].append("merge_commit_has_no_parents")
        except Exception as exc:  # pylint: disable=broad-except
            details["notes"].append(f"merge_commit_lookup_failed:{exc}")

    try:
        commits = gh_api_json(f"repos/{repo}/pulls/{pr_number}/commits?per_page=250")
        if isinstance(commits, list) and commits:
            first = commits[0]
            parents = first.get("parents", [])
            if parents and parents[0].get("sha"):
                details["strategy"] = "first_pr_commit_parent"
                return parents[0]["sha"], details["strategy"], details
            details["notes"].append("first_pr_commit_has_no_parent")
    except Exception as exc:  # pylint: disable=broad-except
        details["notes"].append(f"pull_commit_list_failed:{exc}")

    details["strategy"] = "base_sha_fallback"
    return base_sha, details["strategy"], details


def fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch unified diff text for a PR."""
    return gh_api(
        f"repos/{repo}/pulls/{pr_number}",
        accept="application/vnd.github.v3.diff",
    )


def verify_apply_check(
    repo: str,
    replay_base_sha: str,
    diff_text: str,
    *,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Verify that a PR diff applies cleanly on top of the replay base."""
    with tempfile.TemporaryDirectory(prefix="amdpilot_evals_apply_") as tmp:
        root = Path(tmp)
        clone_dir = root / "repo"
        patch_file = root / "ground_truth.diff"
        patch_file.write_text(diff_text, encoding="utf-8")

        commands = [
            [
                "git",
                "clone",
                "--no-checkout",
                "--filter=blob:none",
                f"https://github.com/{repo}.git",
                str(clone_dir),
            ],
            ["git", "-C", str(clone_dir), "checkout", "--detach", replay_base_sha],
            ["git", "-C", str(clone_dir), "apply", "--check", str(patch_file)],
        ]

        for cmd in commands:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            if proc.returncode != 0:
                return False, f"{' '.join(cmd)}\n{proc.stderr.strip()}"
        return True, "ok"


def manifest_path(path: Path, project_root: Path) -> str:
    """Return a stable path string for JSON outputs.

    Prefer a project-relative path when possible so committed registries remain
    portable. If *path* is outside the repo, fall back to an absolute path.
    """
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
