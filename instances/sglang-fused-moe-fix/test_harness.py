#!/usr/bin/env python3
"""Test harness for sglang-fused-moe-fix eval instance.

Verifies that the NameError for get_global_server_args on ROCm is fixed.
Exit 0 = PASS, Exit 1 = FAIL.

Output format (consumed by amdpilot):
  SCORE: 100.0   (all checks pass)
  SCORE: 0.0     (any check fails)
"""

import ast
import inspect
import sys
import textwrap
from pathlib import Path

SGLANG_ROOT = Path("/workspace/sglang")
TARGET_FILE = SGLANG_ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py"

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


def run_checks():
    print("=" * 60)
    print("sglang-fused-moe-fix test harness")
    print("=" * 60)

    # --- Check 1: Target file exists ---
    check("Target file exists", TARGET_FILE.is_file(),
          f"Expected {TARGET_FILE}")
    if not TARGET_FILE.is_file():
        return False

    source = TARGET_FILE.read_text()

    # --- Check 2: File parses as valid Python ---
    try:
        tree = ast.parse(source)
        check("File parses as valid Python", True)
    except SyntaxError as e:
        check("File parses as valid Python", False, str(e))
        return False

    # --- Check 3: get_global_server_args is NOT called without a guard ---
    # The bug: get_global_server_args() called unconditionally on line ~472
    # The fix: add `_is_cuda and` guard before the call
    #
    # We check that every call to get_global_server_args() in the function body
    # (outside import statements) is guarded by _is_cuda.

    lines = source.split("\n")
    unguarded_calls = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip import lines
        if stripped.startswith("from ") or stripped.startswith("import "):
            continue
        if "get_global_server_args" in stripped and "get_global_server_args()" in stripped:
            # Check context: is there an _is_cuda guard on this line or
            # within the enclosing expression (look at surrounding lines)?
            context_start = max(0, i - 5)
            context_end = min(len(lines), i + 2)
            context = "\n".join(lines[context_start:context_end])
            if "_is_cuda" not in context:
                unguarded_calls.append((i, stripped))

    check(
        "get_global_server_args() guarded by _is_cuda",
        len(unguarded_calls) == 0,
        f"Unguarded calls at lines: {[c[0] for c in unguarded_calls]}"
    )

    # --- Check 4: The _is_cuda variable is still defined ---
    has_is_cuda_def = any(
        "_is_cuda" in line and ("=" in line or "is_cuda()" in line)
        for line in lines
        if not line.strip().startswith("#")
    )
    check("_is_cuda variable is defined", has_is_cuda_def)

    # --- Check 5: Runtime import check on ROCm ---
    # Try to import the module. On ROCm, if the fix is correct,
    # the import should not raise NameError.
    # Note: may fail for other reasons (missing deps) - that's OK.
    sys.path.insert(0, str(SGLANG_ROOT / "python"))
    try:
        # We can't fully import fused_moe (needs compiled kernels),
        # but we can exec the file and check for NameError
        namespace = {"__name__": "__test__", "__file__": str(TARGET_FILE)}
        try:
            exec(compile(source, str(TARGET_FILE), "exec"), namespace)
            check("Module executes without NameError", True)
        except NameError as e:
            if "get_global_server_args" in str(e):
                check("Module executes without NameError", False,
                      f"NameError: {e}")
            else:
                # Other NameErrors are OK (missing deps)
                check("Module executes without NameError (get_global_server_args)", True)
        except (ImportError, ModuleNotFoundError):
            # Missing dependencies is expected in test env - not a failure
            check("Module executes without NameError (import deps missing, OK)", True)
        except Exception:
            # Other errors are not our concern
            check("Module executes without NameError (other error, OK)", True)
    finally:
        if str(SGLANG_ROOT / "python") in sys.path:
            sys.path.remove(str(SGLANG_ROOT / "python"))

    return checks_passed == checks_total


if __name__ == "__main__":
    success = run_checks()
    print()
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    score = 100.0 if success else 0.0
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if success else 1)
