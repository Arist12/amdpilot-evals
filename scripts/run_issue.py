#!/usr/bin/env python3
"""Autonomous issue resolver for amdpilot.

Given a GitHub issue URL, this script:
1. Fetches the issue metadata (title, body, labels, referenced files)
2. Determines the repo, base image, and task type
3. Generates a task description from the issue
4. Generates a YAML config with stages: auto
5. Builds a Docker image with the repo at HEAD
6. Launches amdpilot to resolve the issue autonomously

Usage:
    python scripts/run_issue.py https://github.com/sgl-project/sglang/issues/12345
    python scripts/run_issue.py sgl-project/sglang/issues/12345 --hours 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add the project source to the path so we can use platform detection
sys.path.insert(0, str(PROJECT_ROOT.parent / "src"))


def _get_base_image() -> str:
    """Auto-detect the appropriate base image for the current platform."""
    try:
        from amdpilot.orchestrator.platform import resolve_base_image
        return resolve_base_image()
    except ImportError:
        return "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260306"

WORKSPACE_MAP = {
    "sgl-project/sglang": "/workspace/sglang",
    "ROCm/aiter": "/sgl-workspace/aiter",
    "vllm-project/vllm": "/workspace/vllm",
}


def run_cmd(args: list[str], **kwargs) -> str:
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(args)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return result.stdout.strip()


def parse_issue_url(url: str) -> tuple[str, int]:
    m = re.match(r"(?:https?://github\.com/)?([^/]+/[^/]+?)(?:/issues)?/(\d+)/?$", url)
    if not m:
        raise ValueError(f"Cannot parse issue URL: {url}")
    return m.group(1), int(m.group(2))


def fetch_issue(repo: str, issue_num: int) -> dict:
    raw = run_cmd([
        "gh", "issue", "view", str(issue_num), "-R", repo,
        "--json", "title,body,labels,state"
    ])
    return json.loads(raw)


def classify_issue(data: dict) -> str:
    title = (data.get("title") or "").lower()
    body = (data.get("body") or "").lower()
    labels = [l.get("name", "").lower() for l in (data.get("labels") or [])]
    text = f"{title} {body} {' '.join(labels)}"
    if any(w in text for w in ["bug", "crash", "error", "fix", "regression", "broken"]):
        return "bugfix"
    if any(w in text for w in ["perf", "optimize", "slow", "latency", "throughput"]):
        return "optimize"
    if any(w in text for w in ["feature", "add", "enable", "implement", "support"]):
        return "feature"
    return "bugfix"


def build_task_description(data: dict, repo: str) -> str:
    title = data.get("title", "")
    body = data.get("body", "")
    return textwrap.dedent(f"""\
        # {title}

        ## Issue Description

        {body[:4000] if body else "No description provided."}

        ## Environment

        - Repository: {repo}
        - Docker container with ROCm, PyTorch, AMD GPU
        - Use `/opt/venv/bin/python3` for all commands

        ## Verification

        After applying your fix, run the test harness:
        ```bash
        /opt/venv/bin/python3 /workspace/test_harness.py
        ```
    """)


def build_dockerfile(repo: str, base_image: str) -> str:
    repo_name = repo.split("/")[-1]
    workspace = WORKSPACE_MAP.get(repo, f"/workspace/{repo_name}")

    if repo == "ROCm/aiter":
        return textwrap.dedent(f"""\
            FROM {base_image}
            WORKDIR /workspace
            RUN ln -sf /sgl-workspace/aiter /workspace/aiter
            CMD ["sleep", "infinity"]
        """)

    return textwrap.dedent(f"""\
        FROM {base_image}
        WORKDIR /workspace
        RUN git clone https://github.com/{repo}.git {workspace}
        CMD ["sleep", "infinity"]
    """)


def build_yaml(name: str, repo: str, task_type: str, base_image: str,
               task_desc_path: str, hours: int,
               model_url: str = "") -> str:
    url = model_url or os.environ.get("AMDPILOT_MODEL_URL", "http://localhost:30000/v1")
    return textwrap.dedent(f"""\
        name: {name}
        type: {task_type}
        repo: https://github.com/{repo}.git
        base_image: {name}:base

        model_endpoint:
          base_url: "{url}"
          api_key: "sk-dummy"

        container:
          name: amdpilot_issue_{name.replace("-", "_")}
          gpu: "0"
          shm_size: 16g
          devices: [/dev/kfd, /dev/dri]

        workload:
          description: "Resolve GitHub issue. See task_description.md for details."
          framework: PyTorch

        benchmark:
          command: "/opt/venv/bin/python3 /workspace/test_harness.py"
          metric_name: score
          metric_pattern: 'SCORE:\\s+([\\d.]+)'
          metric_direction: higher

        task:
          description_file: {task_desc_path}

        stages: auto

        kimi_cli:
          repo_url: "https://github.com/Arist12/kimi-cli.git"
          branch: amd-dev
          install_dir: "/sgl-workspace/kimi-cli"
          thinking: true
          yolo: true
          ralph_iterations: -1

        max_retries_per_stage: 3
        max_total_hours: {hours}
    """)


def main():
    parser = argparse.ArgumentParser(description="Resolve a GitHub issue autonomously")
    parser.add_argument("issue", help="Issue URL or owner/repo/issues/NUMBER")
    parser.add_argument("--hours", type=int, default=2, help="Max runtime hours")
    parser.add_argument("--model-url",
                        default=os.environ.get("AMDPILOT_MODEL_URL", ""),
                        help="LLM endpoint URL (or set AMDPILOT_MODEL_URL)")
    parser.add_argument("--results-dir", default=None, help="Results directory")
    parser.add_argument("--dry-run", action="store_true", help="Generate config only")
    args = parser.parse_args()

    repo, issue_num = parse_issue_url(args.issue)
    print(f"Fetching issue #{issue_num} from {repo}...")
    data = fetch_issue(repo, issue_num)

    task_type = classify_issue(data)
    title = data.get("title", "")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:30]
    name = f"issue-{repo.split('/')[-1]}-{issue_num}"

    print(f"  Title: {title}")
    print(f"  Type:  {task_type}")
    print(f"  Name:  {name}")

    base_image = _get_base_image()

    # Create workspace
    work_dir = Path(args.results_dir or f"results/{name}")
    work_dir.mkdir(parents=True, exist_ok=True)

    task_desc = build_task_description(data, repo)
    task_desc_path = work_dir / "task_description.md"
    task_desc_path.write_text(task_desc)

    dockerfile = build_dockerfile(repo, base_image)
    dockerfile_path = work_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile)

    yaml_content = build_yaml(name, repo, task_type, base_image,
                               str(task_desc_path), args.hours,
                               model_url=args.model_url)
    yaml_path = work_dir / "task.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\nGenerated files in {work_dir}:")
    for f in work_dir.iterdir():
        print(f"  {f.name}")

    # Build Docker image
    print(f"\nBuilding Docker image: {name}:base")
    result = subprocess.run(
        ["docker", "build", "-t", f"{name}:base", str(work_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Docker build failed:\n{result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
    print(f"  Image built: {name}:base")

    if args.dry_run:
        print(f"\nDry run — to execute:\n  uv run amdpilot run {yaml_path} --results-dir {work_dir}")
        return

    # Run amdpilot
    print(f"\nLaunching amdpilot (max {args.hours}h)...")
    cmd = [
        "uv", "run", "--project", str(PROJECT_ROOT),
        "amdpilot", "run", str(yaml_path),
        "--results-dir", str(work_dir),
        "--hours", str(args.hours),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # Read results
    trace = work_dir / "trace.md"
    if trace.is_file():
        print(f"\n{'='*60}")
        print(f"TRACE: {trace}")
        print(f"{'='*60}")
        print(trace.read_text())

    # Extract the fix from the committed Docker image and create a git commit
    summary = work_dir / "summary.json"
    if summary.is_file():
        import json as _json
        s = _json.loads(summary.read_text())
        best = s.get("best_metric", "N/A")
        if best != "N/A":
            print(f"\nIssue resolution succeeded (score: {best})")
            print(f"The fix is in the committed Docker image.")
            print(f"To extract and create a PR:")
            print(f"  1. docker run --name tmp {name}:* ls <repo_path>")
            print(f"  2. docker cp tmp:<repo_path> ./fixed-repo")
            print(f"  3. cd fixed-repo && git diff > fix.patch")
            print(f"  4. gh pr create --title 'Fix #{issue_num}' --body 'Auto-generated by amdpilot'")
        else:
            print(f"\nIssue resolution did not produce a passing metric.")


if __name__ == "__main__":
    main()
