#!/usr/bin/env python3
"""Run amdpilot eval instances.

Usage:
    python shared/eval_runner.py instances/sglang-fused-moe-fix/
    python shared/eval_runner.py --list
    python shared/eval_runner.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = EVALS_ROOT / "instances"


def list_instances():
    for instance_dir in sorted(INSTANCES_DIR.iterdir()):
        meta_file = instance_dir / "metadata.json"
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text())
        print(f"  {meta['name']:30s}  [{meta['category']:10s}]  "
              f"difficulty={meta['difficulty']:8s}  "
              f"LOC~{meta.get('expected_loc_changed', '?')}")


def build_docker_image(instance_dir: Path, meta: dict) -> str:
    tag = f"amdpilot-eval-{meta['name']}"
    dockerfile = instance_dir / "Dockerfile"
    if not dockerfile.is_file():
        print(f"  [SKIP] No Dockerfile in {instance_dir}")
        return ""

    print(f"  Building Docker image: {tag}")
    result = subprocess.run(
        ["docker", "build", "-t", tag, str(instance_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [FAIL] Docker build failed:\n{result.stderr[-500:]}")
        return ""

    print(f"  [OK] Image built: {tag}")
    return tag


def run_instance(instance_dir: Path, amdpilot_dir: Path | None = None):
    meta_file = instance_dir / "metadata.json"
    if not meta_file.is_file():
        print(f"Error: {meta_file} not found")
        return False

    meta = json.loads(meta_file.read_text())
    task_yaml = instance_dir / "task.yaml"

    print(f"\n{'='*60}")
    print(f"Running eval: {meta['name']}")
    print(f"Category: {meta['category']} | Difficulty: {meta['difficulty']}")
    print(f"{'='*60}")

    if amdpilot_dir:
        cmd = [
            "uv", "run", "--project", str(amdpilot_dir),
            "amdpilot", "run", str(task_yaml),
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(amdpilot_dir))
        return result.returncode == 0
    else:
        print(f"  Task YAML: {task_yaml}")
        print(f"  To run: uv run amdpilot run {task_yaml}")
        return True


def main():
    parser = argparse.ArgumentParser(description="amdpilot eval runner")
    parser.add_argument("instance", nargs="?", help="Instance directory path")
    parser.add_argument("--list", action="store_true", help="List available instances")
    parser.add_argument("--all", action="store_true", help="Run all instances")
    parser.add_argument("--build-only", action="store_true", help="Only build Docker images")
    parser.add_argument("--amdpilot-dir", help="Path to amdpilot repo")
    args = parser.parse_args()

    if args.list:
        print("Available eval instances:")
        list_instances()
        return

    if args.all:
        for instance_dir in sorted(INSTANCES_DIR.iterdir()):
            if (instance_dir / "metadata.json").is_file():
                if args.build_only:
                    meta = json.loads((instance_dir / "metadata.json").read_text())
                    build_docker_image(instance_dir, meta)
                else:
                    amdpilot = Path(args.amdpilot_dir) if args.amdpilot_dir else None
                    run_instance(instance_dir, amdpilot)
        return

    if args.instance:
        instance_dir = Path(args.instance)
        if not instance_dir.is_dir():
            instance_dir = INSTANCES_DIR / args.instance
        if not instance_dir.is_dir():
            print(f"Error: Instance directory not found: {args.instance}")
            sys.exit(1)

        if args.build_only:
            meta = json.loads((instance_dir / "metadata.json").read_text())
            build_docker_image(instance_dir, meta)
        else:
            amdpilot = Path(args.amdpilot_dir) if args.amdpilot_dir else None
            run_instance(instance_dir, amdpilot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
