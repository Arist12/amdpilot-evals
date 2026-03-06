#!/usr/bin/env python3
"""Semi-automated eval instance curation from a GitHub PR.

Given a merged PR URL, generates a complete eval instance:
  - task_description.md  (symptom only, no solution code)
  - test_harness.py      (LLM-generated or manual)
  - Dockerfile           (from base image template)
  - task.yaml            (stages: auto)
  - metadata.json

Usage:
    python scripts/curate_eval.py --pr https://github.com/sgl-project/sglang/pull/18903
    python scripts/curate_eval.py --pr sgl-project/sglang/18903
    python scripts/curate_eval.py --pr sgl-project/sglang/18903 --generate-test

Data leak prevention:
    - Repo is checked out at merge_commit~1 (the fix does NOT exist)
    - task_description.md describes the symptom/bug, never the solution diff
    - test_harness.py tests expected behavior, not specific code patterns
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent / "instances"

BASE_IMAGES = {
    "sgl-project/sglang": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226",
    "ROCm/aiter": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226",
    "vllm-project/vllm": "rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226",
}

REPO_WORKSPACE_MAP = {
    "sgl-project/sglang": "/workspace/sglang",
    "ROCm/aiter": "/sgl-workspace/aiter",
    "vllm-project/vllm": "/workspace/vllm",
}


def run_gh(args: list[str]) -> str:
    result = subprocess.run(
        ["gh"] + args, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def parse_pr_url(pr_ref: str) -> tuple[str, int]:
    """Parse 'owner/repo/NUMBER' or full GitHub URL into (owner/repo, number)."""
    m = re.match(r"(?:https?://github\.com/)?([^/]+/[^/]+?)(?:/pull)?/(\d+)/?$", pr_ref)
    if not m:
        raise ValueError(f"Cannot parse PR reference: {pr_ref}")
    return m.group(1), int(m.group(2))


def fetch_pr_data(repo: str, pr_num: int) -> dict:
    """Fetch PR metadata via gh CLI."""
    raw = run_gh([
        "pr", "view", str(pr_num), "-R", repo,
        "--json", "title,body,files,mergeCommit,mergedAt,state,labels"
    ])
    data = json.loads(raw)
    if not data.get("mergeCommit"):
        print(f"WARNING: PR #{pr_num} is not merged (state={data.get('state')})")
        print("For unmerged PRs, the 'before' state will use the base branch HEAD.")
    return data


def classify_pr(data: dict) -> str:
    """Infer task type from PR title/body/labels."""
    title = (data.get("title") or "").lower()
    body = (data.get("body") or "").lower()
    labels = [l.get("name", "").lower() for l in (data.get("labels") or [])]
    text = f"{title} {body} {' '.join(labels)}"
    if any(w in text for w in ["fix", "bug", "crash", "error", "regression"]):
        return "bugfix"
    if any(w in text for w in ["optimize", "perf", "speedup", "throughput", "latency"]):
        return "optimize"
    if any(w in text for w in ["feature", "add", "enable", "support", "implement"]):
        return "feature"
    if any(w in text for w in ["port", "rocm", "hip", "amd"]):
        return "fix"
    return "fix"


def estimate_difficulty(data: dict) -> str:
    files = data.get("files") or []
    total_loc = sum(f.get("additions", 0) + f.get("deletions", 0) for f in files)
    n_files = len(files)
    if total_loc <= 10 and n_files <= 1:
        return "easy"
    if total_loc <= 50 and n_files <= 3:
        return "medium"
    if total_loc <= 150:
        return "medium-hard"
    return "hard"


def generate_task_description(data: dict, repo: str) -> str:
    """Generate task_description.md from PR data, stripping solution details."""
    title = data.get("title", "")
    body = data.get("body", "")

    # Strip diff/code blocks that might leak the solution
    body_clean = re.sub(r"```(?:diff|python|bash)?\n.*?```", "[code block removed]",
                        body, flags=re.DOTALL)
    # Strip "## Modifications" / "## Changes" sections that describe the fix
    body_clean = re.sub(
        r"(?:^|\n)##\s*(?:Modifications?|Changes?|Fix|Solution|Implementation)\s*\n.*?(?=\n##|\Z)",
        "", body_clean, flags=re.DOTALL
    )
    body_clean = body_clean.strip()

    files = data.get("files") or []
    affected = [f["path"] for f in files]

    return textwrap.dedent(f"""\
        # {title}

        ## Context

        {body_clean or "See the issue description for details."}

        ## Affected Files

        {chr(10).join(f"- `{f}`" for f in affected)}

        ## Environment

        - Repository: {repo}
        - Docker container with ROCm, PyTorch, AMD GPU
        - Use `/opt/venv/bin/python3` for all commands

        ## Verification

        Run the test harness after applying your fix:
        ```bash
        /opt/venv/bin/python3 /workspace/test_harness.py
        ```
    """)


def generate_dockerfile(repo: str, merge_commit: str | None, base_image: str) -> str:
    """Generate a Dockerfile that checks out the repo at the 'before' state."""
    repo_name = repo.split("/")[-1]
    workspace = REPO_WORKSPACE_MAP.get(repo, f"/workspace/{repo_name}")

    if repo == "ROCm/aiter":
        # aiter is pre-installed in sgl-dev image; checkout the right commit there
        checkout_cmd = ""
        if merge_commit:
            checkout_cmd = (
                f"RUN cd /sgl-workspace/aiter && "
                f"git fetch origin && git checkout {merge_commit}~1"
            )
        return textwrap.dedent(f"""\
            ARG BASE_IMAGE={base_image}
            FROM ${{BASE_IMAGE}}
            WORKDIR /workspace
            {checkout_cmd}
            COPY test_harness.py /workspace/test_harness.py
            COPY task_description.md /workspace/task_description.md
            RUN ln -sf /sgl-workspace/aiter /workspace/aiter
            CMD ["sleep", "infinity"]
        """)

    checkout = f"git checkout {merge_commit}~1" if merge_commit else ""
    return textwrap.dedent(f"""\
        ARG BASE_IMAGE={base_image}
        FROM ${{BASE_IMAGE}}
        WORKDIR /workspace
        RUN git clone --depth 100 https://github.com/{repo}.git {workspace} && \\
            cd {workspace} && {checkout}
        COPY test_harness.py /workspace/test_harness.py
        COPY task_description.md /workspace/task_description.md
        CMD ["sleep", "infinity"]
    """)


def generate_task_yaml(name: str, repo: str, task_type: str,
                       base_image: str,
                       model_url: str = "") -> str:
    url = model_url or os.environ.get("AMDPILOT_MODEL_URL", "http://localhost:30000/v1")
    return textwrap.dedent(f"""\
        name: {name}
        type: {task_type}
        repo: https://github.com/{repo}.git
        base_image: amdpilot-eval-{name}

        model_endpoint:
          model: "qwen-3.5"
          base_url: "{url}"
          api_key: "sk-dummy"

        container:
          name: amdpilot_eval_{name.replace("-", "_")}
          gpu: "0"
          shm_size: 16g
          devices: [/dev/kfd, /dev/dri]

        workload:
          description: "See task_description.md"
          framework: PyTorch

        benchmark:
          command: "/opt/venv/bin/python3 /workspace/test_harness.py"
          metric_name: score
          metric_pattern: 'SCORE:\\s+([\\d.]+)'
          metric_direction: higher

        task:
          description_file: evals/instances/{name}/task_description.md

        stages: auto

        kimi_cli:
          repo_url: "https://github.com/Arist12/kimi-cli.git"
          branch: amd-dev
          install_dir: "/sgl-workspace/kimi-cli"
          thinking: true
          yolo: true
          ralph_iterations: -1

        max_retries_per_stage: 3
        max_total_hours: 2
    """)


def generate_metadata(name: str, repo: str, pr_num: int, data: dict,
                      task_type: str, difficulty: str) -> dict:
    files = data.get("files") or []
    merge = data.get("mergeCommit", {})
    return {
        "name": name,
        "category": task_type,
        "difficulty": difficulty,
        "source": {
            "repo": repo,
            "pr": pr_num,
            "merge_commit": merge.get("oid") if isinstance(merge, dict) else merge,
        },
        "description": data.get("title", ""),
        "expected_loc_changed": sum(f.get("additions", 0) for f in files),
        "expected_files_changed": len(files),
        "affected_files": [f["path"] for f in files],
        "requires_gpu": True,
        "requires_model_download": False,
        "tags": ["rocm", task_type, repo.split("/")[0]],
    }


def generate_test_harness_stub(name: str, data: dict) -> str:
    """Generate a placeholder test harness. Use --generate-test for LLM-generated."""
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Test harness for {name}.

        TODO: This is a stub. Replace with actual verification logic.
        Run the curation script with --generate-test to auto-generate via LLM,
        or write this manually.

        Exit 0 = PASS, Exit 1 = FAIL.
        Output: SCORE: <0-100>
        \"\"\"
        import sys

        checks_passed = 0
        checks_total = 0

        def check(name, condition, detail=""):
            global checks_passed, checks_total
            checks_total += 1
            if condition:
                checks_passed += 1
            status = "PASS" if condition else "FAIL"
            msg = f"  [{{status}}] {{name}}"
            if detail and not condition:
                msg += f": {{detail}}"
            print(msg)
            return condition

        def run_checks():
            print("=" * 60)
            print(f"{name} test harness")
            print("=" * 60)
            # TODO: Add actual checks here
            check("Placeholder", False, "Replace this stub with real tests")

        if __name__ == "__main__":
            run_checks()
            print()
            score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
            print(f"Results: {{checks_passed}}/{{checks_total}} checks passed")
            print(f"SCORE: {{score:.1f}}")
            sys.exit(0 if checks_passed == checks_total else 1)
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Curate an eval instance from a GitHub PR"
    )
    parser.add_argument("--pr", required=True, help="PR URL or owner/repo/NUMBER")
    parser.add_argument("--name", help="Instance name (auto-generated if omitted)")
    parser.add_argument("--generate-test", action="store_true",
                        help="Use LLM to generate test_harness.py (requires model endpoint)")
    parser.add_argument("--model-url",
                        default=os.environ.get("AMDPILOT_MODEL_URL", "http://localhost:30000/v1"),
                        help="LLM endpoint for test generation (or set AMDPILOT_MODEL_URL)")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    repo, pr_num = parse_pr_url(args.pr)
    print(f"Fetching PR #{pr_num} from {repo}...")
    data = fetch_pr_data(repo, pr_num)

    task_type = classify_pr(data)
    difficulty = estimate_difficulty(data)
    title = data.get("title", "")
    merge_commit = data.get("mergeCommit", {})
    if isinstance(merge_commit, dict):
        merge_commit = merge_commit.get("oid")

    # Generate instance name
    name = args.name
    if not name:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        prefix = repo.split("/")[-1]
        name = f"{prefix}-{slug}"

    print(f"  Name:       {name}")
    print(f"  Type:       {task_type}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Commit:     {merge_commit or 'N/A (not merged)'}")
    print(f"  Files:      {len(data.get('files', []))}")

    out_dir = Path(args.output_dir) if args.output_dir else EVALS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_image = BASE_IMAGES.get(repo, BASE_IMAGES["sgl-project/sglang"])

    # Generate all files
    (out_dir / "task_description.md").write_text(
        generate_task_description(data, repo)
    )
    (out_dir / "Dockerfile").write_text(
        generate_dockerfile(repo, merge_commit, base_image)
    )
    (out_dir / "task.yaml").write_text(
        generate_task_yaml(name, repo, task_type, base_image)
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(generate_metadata(name, repo, pr_num, data, task_type, difficulty),
                   indent=2) + "\n"
    )

    if args.generate_test:
        print("  Generating test harness via LLM...")
        from amdpilot.orchestrator.task_analyzer import generate_test_harness
        test_content = generate_test_harness(
            task_description=(out_dir / "task_description.md").read_text(),
            affected_files=[f["path"] for f in (data.get("files") or [])],
            pr_body=data.get("body", ""),
            repo=repo,
            model_url=args.model_url,
        )
        (out_dir / "test_harness.py").write_text(test_content)
    else:
        (out_dir / "test_harness.py").write_text(
            generate_test_harness_stub(name, data)
        )

    print(f"\nEval instance created at: {out_dir}")
    print(f"Files: {', '.join(f.name for f in out_dir.iterdir())}")

    if not args.generate_test:
        print("\nNOTE: test_harness.py is a stub. Either:")
        print("  1. Run with --generate-test to auto-generate via LLM")
        print("  2. Write the test harness manually")

    print(f"\nTo build Docker image: cd {out_dir} && docker build -t amdpilot-eval-{name} .")
    print(f"To run: uv run amdpilot run {out_dir / 'task.yaml'}")


if __name__ == "__main__":
    main()
