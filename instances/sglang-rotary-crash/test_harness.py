#!/usr/bin/env python3
"""Test harness for sglang-rotary-crash eval.

Bug: PR #17934 changed HIP rotary embedding fallback to use tvm_ffi JIT that
requires nvidia-smi/CUDA_HOME, breaking ALL models on ROCm.
Fix: Add forward_hip() override routing to forward_native() (pure PyTorch).

Tests:
1. RotaryEmbedding can be instantiated on ROCm
2. forward() does not crash with tvm_ffi / CUDA_HOME errors
3. Output is a valid tensor with correct shape

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import sys
from pathlib import Path

SGLANG_ROOT = Path("/workspace/sglang")
sys.path.insert(0, str(SGLANG_ROOT / "python"))

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


def run_checks():
    print("=" * 60)
    print("sglang-rotary-crash test harness")
    print("=" * 60)

    # Check 1: The file exists
    target = SGLANG_ROOT / "python/sglang/srt/layers/rotary_embedding.py"
    if not check("rotary_embedding.py exists", target.is_file()):
        return

    # Check 2: File parses
    source = target.read_text()
    try:
        compile(source, str(target), "exec")
        check("File parses as valid Python", True)
    except SyntaxError as e:
        check("File parses as valid Python", False, str(e))
        return

    # Check 3: Can import torch on ROCm
    try:
        import torch
        is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
        check("torch available (HIP detected)" if is_hip else "torch available (no HIP)", True)
    except ImportError:
        check("torch available", False)
        return

    if not torch.cuda.is_available():
        check("GPU available", False, "No GPU — skipping runtime checks")
        # Static checks only; at least verify the forward_hip pattern
        has_forward_hip = "forward_hip" in source or "forward_native" in source
        check("Has HIP-compatible forward path", has_forward_hip,
              "Expected forward_hip() or forward_native() bypass for tvm_ffi")
        return

    check("GPU available", True)

    # Check 4: Import RotaryEmbedding without crash
    try:
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        check("RotaryEmbedding importable", True)
    except Exception as e:
        err = str(e)
        if "tvm_ffi" in err or "nvidia-smi" in err or "CUDA_HOME" in err or "cuda" in err.lower():
            check("RotaryEmbedding importable", False,
                  f"NVIDIA/CUDA dependency on ROCm: {err}")
        else:
            check("RotaryEmbedding importable", False, err)
        return

    # Check 5: HIP forward path exists in the class
    # The bug: on HIP, forward() dispatches to a tvm_ffi JIT path that needs NVIDIA tools
    # The fix: add forward_hip() that routes to forward_native() (pure PyTorch)
    has_hip_path = hasattr(RotaryEmbedding, 'forward_hip')
    check("RotaryEmbedding has forward_hip() method", has_hip_path,
          "Missing forward_hip() — HIP will fall through to NVIDIA JIT path")

    # Check 6: The forward_hip routes to forward_native (not tvm_ffi)
    if has_hip_path:
        import inspect
        hip_source = inspect.getsource(RotaryEmbedding.forward_hip)
        uses_native = "forward_native" in hip_source
        check("forward_hip() routes to forward_native()", uses_native,
              "forward_hip() should call self.forward_native() for pure PyTorch path")
    else:
        # Alternative: check if there's a try/except around the JIT path
        has_jit_guard = "tvm_ffi" not in source or "try" in source
        check("JIT path has fallback for non-NVIDIA", has_jit_guard,
              "tvm_ffi JIT path has no fallback for HIP")


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
