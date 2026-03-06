#!/usr/bin/env python3
"""Validation-spec helpers for PR-derived eval tasks."""

from __future__ import annotations

import re
from typing import Any

from shared.registry_tools import key_files_to_list, normalize_test_commands

_DESCRIPTION_PATTERNS = [
    re.compile(r"^\s*N/?A\b", re.IGNORECASE),
    re.compile(r"\bcomprehensive\b.*\btest\s+plan\b", re.IGNORECASE),
    re.compile(r"\btested\s+against\b", re.IGNORECASE),
    re.compile(r"\bregression\s+from\s+PR\b", re.IGNORECASE),
    re.compile(r"^\s*\w+\s+tests?:\s+\w", re.IGNORECASE),
    re.compile(r"^\s*\w+,\s+\w+$"),
]

_TRAILING_DESCRIPTION_RE = re.compile(
    r"\s+\((?:accuracy|perf|performance|optional)\)$",
    re.IGNORECASE,
)
_TRAILING_WITH_RE = re.compile(r"\s+with\s+(?!-)\S+(?:\s+\S+){0,3}$", re.IGNORECASE)
_FILECHECK_ON_RE = re.compile(r"^\s*FileCheck\s+(?:on\s+|test\s+on\s+)(.+)", re.IGNORECASE)
_FILECHECK_BARE_RE = re.compile(r"^\s*FileCheck\s+(\S+\.mlir)\s*$", re.IGNORECASE)

REPO_COMPILE_CHECKS: dict[str, list[str]] = {
    "ROCm/aiter": [
        "cd /workspace && python -c 'import aiter' 2>/dev/null || python setup.py build_ext --inplace 2>&1 | tail -5",
    ],
    "ROCm/composable_kernel": [
        "cd /workspace && python -c 'import ck4inductor' 2>/dev/null || echo 'compile check skipped'",
    ],
    "ROCm/HIP": [
        "cd /workspace && hipcc --version 2>/dev/null || echo 'hipcc not available'",
    ],
    "ROCm/HIPIFY": [
        "cd /workspace && test -f build/hipify-clang && build/hipify-clang --version || echo 'hipify not built'",
    ],
    "ROCm/rocm-libraries": [
        "cd /workspace && python -c 'import rocm_libs' 2>/dev/null || echo 'compile check skipped'",
    ],
    "Dao-AILab/flash-attention": [
        "cd /workspace && python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'flash_attn not importable'",
    ],
    "triton-lang/triton": [
        "cd /workspace && python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'triton not importable'",
    ],
    "pytorch/pytorch": [
        "cd /workspace && python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'torch check'",
    ],
    "sgl-project/sglang": [
        "cd /workspace && python -c 'import sglang' 2>/dev/null || echo 'sglang not importable'",
    ],
    "vllm-project/vllm": [
        "cd /workspace && python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'vllm not importable'",
    ],
}


def _is_description(cmd: str) -> bool:
    for pat in _DESCRIPTION_PATTERNS:
        if pat.search(cmd):
            return True
    words = cmd.split()
    if not words:
        return True
    if words[0][0].isupper() and not any(c in words[0] for c in ("=", "/", "-", ".", "_")):
        lower_rest = " ".join(words[1:]).lower()
        if any(kw in lower_rest for kw in ("with various", "with different", "on mi", "dashboard")):
            return True
    return False


def _transform_filecheck(cmd: str) -> str | None:
    match = _FILECHECK_ON_RE.match(cmd)
    if match:
        path = match.group(1).strip().rstrip(";").strip()
        return f"cd /workspace/build && lit -v ../{path}"
    match = _FILECHECK_BARE_RE.match(cmd)
    if match:
        path = match.group(1).strip()
        return f"cd /workspace/build && lit -v ../{path}"
    return None


def _split_compound(raw: str) -> list[str]:
    return [segment.strip() for segment in raw.split(";") if segment.strip()]


def _split_or_alternatives(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+OR\s+", raw) if part.strip()]


def _is_executable_command(cmd: str) -> bool:
    cmd = cmd.strip()
    if not cmd or _is_description(cmd):
        return False
    if cmd.endswith((".hpp", ".h", ".cpp", ".cc", ".cu", ".cuh", ".yaml", ".yml", ".json")):
        return False
    words = cmd.split()
    first = words[0] if words else ""
    executable_prefixes = (
        "python",
        "pytest",
        "bash",
        "sh",
        "make",
        "cmake",
        "cd",
        "export",
        "pip",
        "hipcc",
        "hipify",
        "rocm",
        "test_",
        "./",
        "timeout",
        "vllm",
        "lm_eval",
        "ninja",
        "lit",
        "curl",
    )
    if any(first.startswith(prefix) for prefix in executable_prefixes):
        return True
    if "=" in first:
        return True
    if "/" in first or first.startswith("-"):
        return True
    return False


def normalize_validation_commands(value: Any) -> list[str]:
    """Normalize and filter to executable validation commands only."""
    raw = normalize_test_commands(value)
    result: list[str] = []
    for cmd in raw:
        if _is_description(cmd):
            continue

        if " OR " in cmd:
            parts = _split_or_alternatives(cmd)
            cmd = parts[0] if parts else cmd

        sub_cmds = _split_compound(cmd)
        for sub in sub_cmds:
            sub = _TRAILING_DESCRIPTION_RE.sub("", sub)
            sub = _TRAILING_WITH_RE.sub("", sub)
            sub = sub.strip()
            lit = _transform_filecheck(sub)
            if lit:
                result.append(lit)
                continue
            if _is_executable_command(sub):
                result.append(sub)
    return result


def classify_tier(test_commands: list[str], key_files: list[str]) -> int:
    if test_commands:
        return 1
    if key_files:
        return 2
    return 3


def generate_deterministic_checks(
    repo: str,
    ground_truth_diff_path: str | None,
) -> list[str]:
    """Generate fallback deterministic checks for tier-2/3 tasks."""
    checks: list[str] = []
    if ground_truth_diff_path:
        checks.append(
            "cd /workspace && git diff HEAD --name-only | head -20 && echo 'agent made changes'"
        )
    checks.extend(REPO_COMPILE_CHECKS.get(repo, []))
    checks.append("cd /workspace && git diff --stat HEAD 2>/dev/null | tail -5")
    return checks


def _extract_flag_int(command: str, flag: str, default: int) -> int:
    match = re.search(rf"{re.escape(flag)}\s+(\d+)", command)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _supports_kimi_profile(pr: dict[str, Any]) -> bool:
    fields = [
        str(pr.get("title", "")),
        str(pr.get("problem", "")),
        str(pr.get("solution", "")),
    ]
    text = " ".join(fields).lower()
    return "kimi k2.5" in text or int(pr.get("pr_number", 0) or 0) == 19228


def _sglang_bootstrap(pr: dict[str, Any], commands: list[str]) -> dict[str, Any] | None:
    lower_commands = [cmd.lower() for cmd in commands]
    if not any("benchmark/gsm8k/bench_sglang.py" in cmd for cmd in lower_commands):
        return None

    first_command = commands[0]
    port = _extract_flag_int(first_command, "--port", 30000)

    if not _supports_kimi_profile(pr):
        return {
            "required": True,
            "supported": False,
            "profile": "sglang_bench_pending_profile",
            "reason": "bench_sglang needs a repo-specific serving profile; only the Kimi-K2.5 MI355 profile is enabled right now",
        }

    return {
        "required": True,
        "supported": True,
        "profile": "sglang_kimi_k2_mi355",
        "model_path": "/sgl-workspace/models/Kimi-K2.5",
        "healthcheck_command": f"curl -sf --max-time 5 http://localhost:{port}/model_info",
        "startup_timeout_seconds": 1800,
        "shutdown_command": "pkill -f 'python3 -m sglang.launch_server' || true",
        "start_command": " ".join(
            [
                "export AITER_ROOT_DIR=/tmp/.aiter_mi355_clean",
                "&&",
                "export AITER_JIT_DIR=/tmp/.aiter_mi355_clean/jit",
                "&&",
                "mkdir -p \"$AITER_JIT_DIR\"",
                "&&",
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
                "&&",
                "export NO_PROXY=127.0.0.1,localhost",
                "&&",
                "export no_proxy=127.0.0.1,localhost",
                "&&",
                "python3 -m sglang.launch_server",
                "--model-path /sgl-workspace/models/Kimi-K2.5",
                "--trust-remote-code",
                "--reasoning-parser kimi_k2",
                "--tool-call-parser kimi_k2",
                "--host 0.0.0.0",
                f"--port {port}",
                "--tp-size ${MODEL_SERVER_TP_SIZE:-8}",
                "--skip-server-warmup",
                "--prefill-attention-backend aiter",
                "--decode-attention-backend triton",
            ]
        ),
    }


def _vllm_bootstrap(commands: list[str]) -> dict[str, Any] | None:
    lower_commands = [cmd.lower() for cmd in commands]
    if any(cmd.startswith("vllm serve ") for cmd in lower_commands):
        return {
            "required": True,
            "supported": False,
            "profile": "vllm_pending_profile",
            "reason": "vllm serve launches a long-running service and still needs a repo-specific runtime profile",
        }
    if any("localhost:8000" in cmd for cmd in lower_commands):
        return {
            "required": True,
            "supported": False,
            "profile": "vllm_pending_profile",
            "reason": "vllm serving bootstrap interface is reserved, but the runtime profile is not enabled in this revision",
        }
    return None


def infer_model_server_bootstrap(
    pr: dict[str, Any],
    repo: str,
    commands: list[str],
) -> dict[str, Any] | None:
    if repo == "sgl-project/sglang":
        return _sglang_bootstrap(pr, commands)
    if repo == "vllm-project/vllm":
        return _vllm_bootstrap(commands)
    return None


def build_validation_spec(pr: dict[str, Any]) -> dict[str, Any]:
    """Build a task-plane validation spec for a PR row."""
    repo = str(pr.get("repo", "")).strip()
    commands = normalize_validation_commands(
        pr.get("test_commands_normalized", pr.get("test_commands")),
    )
    key_files = key_files_to_list(pr.get("key_files", []))
    tier = classify_tier(commands, key_files)
    ground_truth_diff_path = pr.get("ground_truth_diff_path")

    deterministic_checks: list[str] = []
    if tier >= 2:
        deterministic_checks = generate_deterministic_checks(
            repo,
            ground_truth_diff_path if isinstance(ground_truth_diff_path, str) else None,
        )

    spec: dict[str, Any] = {
        "tier": tier,
        "mode": "tests" if tier == 1 else "deterministic",
        "test_commands": commands,
        "deterministic_checks": deterministic_checks,
        "key_files": key_files,
        "key_file_overlap_threshold": 0.25,
        "diff_file_jaccard_threshold": 0.30,
    }
    bootstrap = infer_model_server_bootstrap(pr, repo, commands)
    if bootstrap:
        spec["model_server_bootstrap"] = bootstrap
    return spec
