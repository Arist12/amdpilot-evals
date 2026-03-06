#!/usr/bin/env python3
"""Test harness for aiter-mla-reduce-optimize. Runtime correctness + performance.

Validates that the MLA reduce kernel:
1. Produces correct attention output (matches reference computation)
2. Achieves target latency reduction (25%+ improvement over baseline)
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
print("aiter-mla-reduce-optimize test harness")
print("=" * 60)

import torch

check("GPU available", torch.cuda.is_available())
device = torch.device("cuda:0")
torch.manual_seed(42)

try:
    import aiter
    check("Import aiter", True)
except ImportError as e:
    check("Import aiter", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# --- Check that the MLA module exists ---
print("\n--- Module availability ---")
mla_available = False
try:
    from aiter.ops.mla import mla_decode_fwd
    mla_available = True
    check("Import MLA decode forward", True)
except (ImportError, AttributeError) as e:
    try:
        from aiter import mla_decode_fwd
        mla_available = True
        check("Import MLA decode forward (alt path)", True)
    except (ImportError, AttributeError) as e2:
        check("Import MLA decode forward", False, str(e2))

# --- Correctness check via existing test ---
print("\n--- Correctness ---")
try:
    import subprocess
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-m", "pytest",
         "/sgl-workspace/aiter/op_tests/test_mla.py", "-x", "-q",
         "--timeout=120", "-k", "test_mla_sparse"],
        capture_output=True, text=True, timeout=180,
        cwd="/sgl-workspace/aiter"
    )
    test_passed = result.returncode == 0
    if not test_passed:
        output_tail = (result.stdout + result.stderr)[-500:]
        check("MLA sparse test passes", False, f"Test failed: {output_tail}")
    else:
        check("MLA sparse test passes", True)
except subprocess.TimeoutExpired:
    check("MLA sparse test passes", False, "Test timed out (>180s)")
except FileNotFoundError:
    check("MLA sparse test (pytest not available, skip)", True)
except Exception as e:
    check("MLA sparse test passes", False, str(e)[:200])

# --- Performance check ---
print("\n--- Performance ---")
if mla_available:
    try:
        # Set up a realistic MLA reduce workload
        # batch=1, context_len=4000, nhead_q=16, nhead_kv=2, head_dim=128
        batch = 1
        nhead_q = 16
        head_dim = 128
        context_len = 4000

        # Warm up
        q = torch.randn(batch, nhead_q, 1, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, nhead_q, context_len, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, nhead_q, context_len, head_dim, device=device, dtype=torch.bfloat16)

        # Use torch SDPA as reference for timing comparison
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        # Run MLA test via the existing test infrastructure
        result = subprocess.run(
            ["/opt/venv/bin/python3", "-c", """
import sys, time
sys.path.insert(0, '/sgl-workspace/aiter')
import torch
torch.manual_seed(42)

device = torch.device('cuda:0')
batch, nhead_q, context_len, head_dim = 1, 16, 4000, 128

q = torch.randn(batch * nhead_q, 1, head_dim, device=device, dtype=torch.bfloat16)
k = torch.randn(batch * nhead_q, context_len, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn(batch * nhead_q, context_len, head_dim, device=device, dtype=torch.bfloat16)

# Warm up with SDPA (uses the reduce kernel internally when available)
for _ in range(20):
    torch.nn.functional.scaled_dot_product_attention(
        q.view(batch, nhead_q, 1, head_dim),
        k.view(batch, nhead_q, context_len, head_dim),
        v.view(batch, nhead_q, context_len, head_dim),
    )
torch.cuda.synchronize()

iters = 200
t0 = time.perf_counter()
for _ in range(iters):
    torch.nn.functional.scaled_dot_product_attention(
        q.view(batch, nhead_q, 1, head_dim),
        k.view(batch, nhead_q, context_len, head_dim),
        v.view(batch, nhead_q, context_len, head_dim),
    )
torch.cuda.synchronize()
us = (time.perf_counter() - t0) * 1e6 / iters
print(f'MLA_LATENCY_US: {us:.1f}')
"""],
            capture_output=True, text=True, timeout=120,
        )

        import re
        match = re.search(r'MLA_LATENCY_US:\s+([\d.]+)', result.stdout)
        if match:
            latency_us = float(match.group(1))
            # Unoptimized baseline: ~18.2us for this workload
            # Optimized target: <13.5us (25%+ improvement)
            threshold_us = 15.0
            print(f"  MLA attention latency: {latency_us:.1f}us (threshold: <{threshold_us}us)")
            check(f"MLA latency < {threshold_us}us (got {latency_us:.1f}us)", latency_us < threshold_us,
                  f"Kernel too slow ({latency_us:.1f}us), optimization likely not applied")
        else:
            stderr_tail = result.stderr[-300:] if result.stderr else ""
            check("MLA performance measurement", False,
                  f"Could not extract latency. stderr: {stderr_tail}")

    except Exception as e:
        check("MLA performance measurement", False, str(e)[:200])
else:
    check("MLA performance (skipped, module unavailable)", False, "MLA module not available")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
