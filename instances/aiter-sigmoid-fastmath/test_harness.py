#!/usr/bin/env python3
"""Test harness for aiter-sigmoid-fastmath. Runtime correctness + performance.

Validates that the sigmoid activation kernel:
1. Produces correct results (matches PyTorch reference)
2. Achieves at least 15% speedup over the baseline
"""
import sys
import time

sys.path.insert(0, "/sgl-workspace/aiter")

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


print("=" * 60)
print("aiter-sigmoid-fastmath test harness")
print("=" * 60)

import torch

check("GPU available", torch.cuda.is_available())
device = torch.device("cuda:0")
torch.manual_seed(42)

try:
    from aiter.ops.aiter_operator import sigmoid
    check("Import aiter sigmoid", True)
except ImportError as e:
    check("Import aiter sigmoid", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# --- Correctness checks ---
print("\n--- Correctness ---")

for dtype_name, dtype, atol in [("float16", torch.float16, 1e-3), ("bfloat16", torch.bfloat16, 1e-2)]:
    x = torch.randn(1024, 2048, device=device, dtype=dtype)
    ref = torch.sigmoid(x)
    out = sigmoid(x)
    max_diff = (out - ref).abs().max().item()
    check(f"Sigmoid correctness ({dtype_name}, max_diff={max_diff:.2e})", max_diff < atol,
          f"max_diff={max_diff:.2e} exceeds {atol}")

# Edge cases: very large and very small values (use 2D tensor for kernel compatibility)
x_extreme = torch.tensor([[0.0, 1e-6, -1e-6, 50.0, -50.0, 100.0, -100.0]],
                         device=device, dtype=torch.float16)
out_extreme = sigmoid(x_extreme)
ref_extreme = torch.sigmoid(x_extreme)
max_diff_edge = (out_extreme - ref_extreme).abs().max().item()
check(f"Edge case correctness (max_diff={max_diff_edge:.2e})", max_diff_edge < 1e-2)

# Also test silu_and_mul if available (uses sigmoid internally)
try:
    from aiter.ops.activation import silu_and_mul
    x_silu = torch.randn(1024, 4096, device=device, dtype=torch.bfloat16)
    out_silu = torch.empty(1024, 2048, device=device, dtype=torch.bfloat16)
    silu_and_mul(out_silu, x_silu)
    x_a, x_b = x_silu.chunk(2, dim=-1)
    ref_silu = torch.nn.functional.silu(x_a) * x_b
    max_diff_silu = (out_silu - ref_silu).abs().max().item()
    check(f"SiLU_and_mul correctness (max_diff={max_diff_silu:.2e})", max_diff_silu < 0.1)
except ImportError:
    check("SiLU_and_mul correctness (skipped, not available)", True)

# --- Performance checks ---
print("\n--- Performance ---")

sizes = [(4096, 4096, torch.bfloat16)]
for M, N, dtype in sizes:
    x = torch.randn(M, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(30):
        sigmoid(x)
    torch.cuda.synchronize()

    # Benchmark
    iters = 500
    t0 = time.perf_counter()
    for _ in range(iters):
        sigmoid(x)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iters

    # Unoptimized baseline: ~22-26us for 4096x4096 bf16
    # Optimized with AMD intrinsics: ~14-16us (>30% improvement)
    # Threshold set at 18us to verify optimization is applied
    threshold_us = 18.0
    print(f"  Sigmoid latency ({M}x{N} {dtype}): {us:.1f}us (threshold: <{threshold_us}us)")
    check(f"Sigmoid latency < {threshold_us}us (got {us:.1f}us)", us < threshold_us,
          f"Kernel too slow ({us:.1f}us), optimization likely not applied")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
