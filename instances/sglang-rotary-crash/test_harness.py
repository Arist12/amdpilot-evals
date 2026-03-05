#!/usr/bin/env python3
"""Test harness for sglang-rotary-crash.

Verifies RotaryEmbedding works on ROCm without requiring CUDA_HOME.
All checks are RUNTIME -- no static string matching.

Exit 0 = PASS, Exit 1 = FAIL.
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

# Check 1: Import RotaryEmbedding without CUDA_HOME crash
try:
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    check("Import RotaryEmbedding", True)
except RuntimeError as e:
    if "CUDA" in str(e) or "cuda" in str(e):
        check("Import RotaryEmbedding", False,
              f"CUDA dependency on ROCm: {e}")
    else:
        check("Import RotaryEmbedding", False, str(e))
except Exception as e:
    check("Import RotaryEmbedding", False, str(e))

# Check 2: Instantiate RotaryEmbedding
try:
    import torch
    rope = RotaryEmbedding(
        head_size=128, rotary_dim=128,
        max_position_embeddings=4096, base=10000,
        is_neox_style=True, dtype=torch.bfloat16,
    )
    check("Instantiate RotaryEmbedding", True)
except RuntimeError as e:
    if "CUDA" in str(e):
        check("Instantiate RotaryEmbedding", False, f"CUDA error: {e}")
    else:
        check("Instantiate RotaryEmbedding", False, str(e))
except Exception as e:
    check("Instantiate RotaryEmbedding", False, str(e))

# Check 3: Call the forward path that is dispatched on HIP
# This is the critical check -- on unfixed code, this triggers
# the tvm_ffi / CUDA_HOME error path
try:
    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_hip and torch.cuda.is_available():
        device = torch.device("cuda:0")
        seq_len, num_heads, head_dim = 32, 8, 128
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        q = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.bfloat16)

        # This calls the dispatch chain that broke on ROCm
        q_out, k_out = rope(positions, q, k)

        has_nan = torch.isnan(q_out).any().item() or torch.isnan(k_out).any().item()
        check("Forward pass on HIP (no CUDA_HOME crash)", not has_nan,
              "Output contains NaN" if has_nan else "")
    else:
        check("Forward pass on HIP", False, "Not running on HIP or no GPU")
except RuntimeError as e:
    err = str(e)
    if "CUDA" in err or "tvm" in err or "nvidia" in err.lower():
        check("Forward pass on HIP", False, f"CUDA/TVM dependency: {err}")
    else:
        check("Forward pass on HIP", False, err)
except Exception as e:
    check("Forward pass on HIP", False, str(e))

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
