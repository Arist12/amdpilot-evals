#!/usr/bin/env python3
"""Test harness for sglang-deepseek-r1-optimize eval.

Bug/Perf: DeepSeek-V3/R1 optimized code paths restricted to gfx950 only.
The fused_qk_rope_cat_and_cache_mla kernel (line ~1783-1791) is gated behind
_use_aiter_gfx95 instead of _use_aiter, blocking MI300X from using it.

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import sys
from pathlib import Path

SGLANG_ROOT = Path("/workspace/sglang")

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


def check_fused_kernel_not_gfx95_only():
    """The fused RoPE+cat+cache MLA kernel should be available on ALL HIP, not just gfx950."""
    deepseek = SGLANG_ROOT / "python/sglang/srt/models/deepseek_v2.py"
    if not check("deepseek_v2.py exists", deepseek.is_file()):
        return

    source = deepseek.read_text()
    lines = source.split("\n")

    # Find where fused_qk_rope_cat_and_cache_mla is called
    # The bug: it's inside an `if _use_aiter_gfx95:` block
    # The fix: should be inside `if _use_aiter:` (all HIP) instead
    fused_rope_in_gfx95_block = False
    for i, line in enumerate(lines):
        if "fused_qk_rope_cat_and_cache_mla" in line and "import" not in line:
            # Look at the enclosing if-block (search backwards for nearest if)
            for j in range(i - 1, max(0, i - 15), -1):
                stripped = lines[j].strip()
                if stripped.startswith("if ") and ":" in stripped:
                    if "_use_aiter_gfx95" in stripped:
                        fused_rope_in_gfx95_block = True
                    break

    check("fused_qk_rope_cat_and_cache_mla available on all HIP (not gfx950-only)",
          not fused_rope_in_gfx95_block,
          "fused_qk_rope_cat_and_cache_mla is gated behind _use_aiter_gfx95")


def check_fp8_gemm_broadened():
    """Check that FP8 quantization code paths are not gfx950-restricted."""
    fp8_utils = SGLANG_ROOT / "python/sglang/srt/layers/quantization/fp8_utils.py"
    if not check("fp8_utils.py exists", fp8_utils.is_file()):
        return

    source = fp8_utils.read_text()
    # The fix should make FP8 GEMM configs available beyond gfx950
    # Check for presence of gfx942-compatible config or broadened conditions
    has_broadened_fp8 = ("gfx942" in source or
                         "is_hip" in source.lower() or
                         "_use_aiter" in source)
    check("FP8 GEMM paths accessible beyond gfx950", has_broadened_fp8)


def check_aiter_backend_broadened():
    """Check that the aiter attention backend paths are broadened."""
    aiter_backend = SGLANG_ROOT / "python/sglang/srt/layers/attention/aiter_backend.py"
    if not check("aiter_backend.py exists", aiter_backend.is_file()):
        return

    source = aiter_backend.read_text()
    lines = source.split("\n")

    # The fix may broaden conditions in the attention backend
    # Count how many times _use_aiter_gfx95 (restrictive) vs _use_aiter (broad) is used
    gfx95_refs = sum(1 for l in lines if "_use_aiter_gfx95" in l and "import" not in l)
    aiter_refs = sum(1 for l in lines
                     if "_use_aiter" in l and "_use_aiter_gfx95" not in l
                     and "import" not in l)

    check(f"aiter_backend uses broad _use_aiter paths (gfx95-only refs: {gfx95_refs})",
          gfx95_refs <= 2,
          f"Too many gfx950-only gates ({gfx95_refs}); should use _use_aiter for most paths")


def run_checks():
    print("=" * 60)
    print("sglang-deepseek-r1-optimize test harness")
    print("=" * 60)

    print("\n--- Check 1: Fused kernel dispatch ---")
    check_fused_kernel_not_gfx95_only()

    print("\n--- Check 2: FP8 GEMM configs ---")
    check_fp8_gemm_broadened()

    print("\n--- Check 3: Aiter backend paths ---")
    check_aiter_backend_broadened()


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
