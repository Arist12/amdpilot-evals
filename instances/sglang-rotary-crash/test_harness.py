#!/usr/bin/env python3
"""Test harness for sglang-rotary-crash. ALL RUNTIME CHECKS.

Bug: RotaryEmbedding forward() on HIP tries to JIT-compile CUDA code,
crashing with CUDA_HOME / hipcc errors.
"""
import sys
sys.path.insert(0, "/workspace/sglang/python")

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

print("=" * 60)
print("sglang-rotary-crash test harness")
print("=" * 60)

# Mock server args (needed to instantiate RotaryEmbedding)
try:
    from sglang.srt.server_args import ServerArgs
    import sglang.srt.server_args as sa
    sa._global_server_args = ServerArgs(model_path="dummy")
except Exception as e:
    print(f"Warning: Could not mock server args: {e}")

# Check 1: Import
try:
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    check("Import RotaryEmbedding", True)
except Exception as e:
    check("Import RotaryEmbedding", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 2: Instantiate
try:
    import torch
    rope = RotaryEmbedding(128, 128, 4096, 10000, True, torch.bfloat16)
    check("Instantiate RotaryEmbedding", True)
except Exception as e:
    check("Instantiate RotaryEmbedding", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 3: Forward pass on HIP -- this is where the bug manifests
is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
if is_hip and torch.cuda.is_available():
    try:
        device = torch.device("cuda:0")
        positions = torch.arange(32, device=device).unsqueeze(0)
        q = torch.randn(32, 8 * 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(32, 8 * 128, device=device, dtype=torch.bfloat16)

        q_out, k_out = rope(positions, q, k)

        has_nan = torch.isnan(q_out).any().item() or torch.isnan(k_out).any().item()
        check("Forward pass on HIP (no CUDA JIT crash)", not has_nan,
              "Output contains NaN" if has_nan else "")
    except RuntimeError as e:
        err = str(e)
        if "ninja" in err or "CUDA" in err or "hipcc" in err or "expt-relaxed" in err:
            check("Forward pass on HIP (no CUDA JIT crash)", False,
                  f"JIT compilation failure: {err[:200]}")
        else:
            check("Forward pass on HIP", False, err[:200])
    except Exception as e:
        check("Forward pass on HIP", False, str(e)[:200])
else:
    check("Forward pass on HIP", False, "No HIP GPU available")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
