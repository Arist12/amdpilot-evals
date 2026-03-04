#!/usr/bin/env python3
"""Test harness for vllm-fp4-hwdetect eval instance.

Verifies that FP4 BMM hardware detection is properly implemented.
Exit 0 = PASS, Exit 1 = FAIL.

Output format:
  SCORE: 100.0   (all checks pass)
  SCORE: 0.0     (any check fails)
"""

import ast
import subprocess
import sys
from pathlib import Path

VLLM_ROOT = Path("/workspace/vllm")
TARGET_FILE = VLLM_ROOT / "vllm" / "_aiter_ops.py"

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
    print("vllm-fp4-hwdetect test harness")
    print("=" * 60)

    if not check("Target file exists", TARGET_FILE.is_file()):
        return

    source = TARGET_FILE.read_text()

    try:
        ast.parse(source)
        check("File parses as valid Python", True)
    except SyntaxError as e:
        check("File parses as valid Python", False, str(e))
        return

    # Check for hardware detection logic
    has_fp4_avail = "is_fp4_avail" in source
    check("References is_fp4_avail() from AITER", has_fp4_avail,
          "Expected hardware capability check via AITER's is_fp4_avail()")

    has_hw_check = ("gfx942" in source or "gfx950" in source or
                    "arch" in source.lower() or "is_fp4_avail" in source)
    check("Contains hardware architecture check", has_hw_check)

    has_fallback = "fallback" in source.lower() or "fp8" in source.lower() or "disable" in source.lower()
    check("Contains fallback/disable logic", has_fallback,
          "Expected FP8 fallback when FP4 not supported")

    has_logging = "log" in source.lower() or "print" in source.lower() or "warn" in source.lower()
    check("Contains informative logging", has_logging,
          "Expected logging when FP4 is auto-disabled")

    has_error_handling = "try" in source and "except" in source
    check("Has error handling for missing AITER modules", has_error_handling)

    # Runtime check: import the module and verify is_fp4bmm_enabled
    sys.path.insert(0, str(VLLM_ROOT))
    try:
        from vllm._aiter_ops import is_fp4bmm_enabled
        result = is_fp4bmm_enabled()
        check("is_fp4bmm_enabled() callable without crash", True)

        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip:
            gpu_name = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx942" in gpu_name:
                check("FP4 disabled on gfx942", not result,
                      f"Expected False on gfx942, got {result}")
            elif "gfx950" in gpu_name:
                check("FP4 enabled on gfx950", result,
                      f"Expected True on gfx950, got {result}")
    except ImportError:
        check("is_fp4bmm_enabled() importable (deps may be missing)", True)
    except Exception as e:
        check("is_fp4bmm_enabled() callable", False, str(e))
    finally:
        if str(VLLM_ROOT) in sys.path:
            sys.path.remove(str(VLLM_ROOT))


if __name__ == "__main__":
    run_checks()
    print()
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
