#!/usr/bin/env python3
"""Test harness for aiter-mla-reduce-optimize. Runtime correctness + performance.

Validates that the MLA reduce kernel:
1. Produces correct attention output (matches reference computation)
2. Achieves target latency reduction in persistent-mode MLA decode
"""
import sys
import time
import subprocess
import re

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

# --- Module availability ---
print("\n--- Module availability ---")
mla_available = False
try:
    from aiter.mla import mla_decode_fwd
    mla_available = True
    check("Import MLA decode forward", True)
except (ImportError, AttributeError) as e:
    try:
        mla_decode_fwd = aiter.mla.mla_decode_fwd
        mla_available = True
        check("Import MLA decode forward (alt path)", True)
    except (ImportError, AttributeError) as e2:
        check("Import MLA decode forward", False, str(e2))

# --- Correctness check via existing test ---
print("\n--- Correctness ---")
try:
    result = subprocess.run(
        ["/opt/venv/bin/python3", "/sgl-workspace/aiter/op_tests/test_mla.py",
         "-n", "16,1", "-b", "1", "-c", "4000", "-d", "bf16", "-kvd", "bf16"],
        capture_output=True, text=True, timeout=300,
        cwd="/sgl-workspace/aiter"
    )
    output = result.stdout + result.stderr
    passed = "passed" in output.lower() and result.returncode == 0
    if not passed:
        output_tail = output[-500:]
        check("MLA decode correctness test", False, f"Test failed: {output_tail}")
    else:
        check("MLA decode correctness test", True)
except subprocess.TimeoutExpired:
    check("MLA decode correctness test", False, "Test timed out (>300s)")
except Exception as e:
    check("MLA decode correctness test", False, str(e)[:200])

# --- Performance check: persistent-mode MLA decode (exercises reduce.cu) ---
print("\n--- Performance ---")
if mla_available:
    try:
        result = subprocess.run(
            ["/opt/venv/bin/python3", "-c", r"""
import sys, time, math
sys.path.insert(0, '/sgl-workspace/aiter')
import torch
import aiter
from aiter import dtypes
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

device = torch.device('cuda:0')
torch.manual_seed(42)

batch_size = 64
max_seqlen_qo = 1
nhead, nhead_kv = 16, 1
kv_lora_rank, qk_rope_head_dim = 512, 64
qk_head_dim = kv_lora_rank + qk_rope_head_dim
v_head_dim = kv_lora_rank
page_size = 1
sm_scale = 1.0 / (qk_head_dim ** 0.5)
ctx_len = 4000

total_q = batch_size * max_seqlen_qo
num_page = ctx_len * batch_size

qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
for i in range(batch_size):
    qo_indptr[i+1] = qo_indptr[i] + max_seqlen_qo
    kv_indptr[i+1] = kv_indptr[i] + ctx_len
kv_indices = torch.arange(num_page, dtype=torch.int, device=device)
kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device=device)

q = torch.randn(total_q, nhead, qk_head_dim, dtype=torch.bfloat16, device=device)
kv_buf = torch.randn(num_page, page_size, nhead_kv, qk_head_dim, dtype=torch.bfloat16, device=device)
o = torch.empty(total_q, nhead, v_head_dim, dtype=torch.bfloat16, device=device)

info = get_mla_metadata_info_v1(
    batch_size, max_seqlen_qo, nhead,
    torch.bfloat16, torch.bfloat16, is_sparse=False, fast_mode=True)
wmp = torch.empty(info[0][0], dtype=info[0][1], device=device)
wi  = torch.empty(info[1][0], dtype=info[1][1], device=device)
wis = torch.empty(info[2][0], dtype=info[2][1], device=device)
ri  = torch.empty(info[3][0], dtype=info[3][1], device=device)
rfm = torch.empty(info[4][0], dtype=info[4][1], device=device)
rpm = torch.empty(info[5][0], dtype=info[5][1], device=device)

get_mla_metadata_v1(
    qo_indptr, kv_indptr, kv_last_page_lens,
    nhead // nhead_kv, nhead_kv, False,
    wmp, wis, wi, ri, rfm, rpm, page_size, 16)

def run_decode():
    aiter.mla.mla_decode_fwd(
        q, kv_buf, o, qo_indptr, kv_indptr, kv_indices,
        kv_last_page_lens, max_seqlen_qo, page_size, nhead_kv, sm_scale,
        work_meta_data=wmp, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm)

for _ in range(20):
    run_decode()
torch.cuda.synchronize()

iters = 200
t0 = time.perf_counter()
for _ in range(iters):
    run_decode()
torch.cuda.synchronize()
us = (time.perf_counter() - t0) * 1e6 / iters
print(f'MLA_PERSISTENT_DECODE_US: {us:.1f}')
"""],
            capture_output=True, text=True, timeout=180,
        )

        match = re.search(r'MLA_PERSISTENT_DECODE_US:\s+([\d.]+)', result.stdout)
        if match:
            latency_us = float(match.group(1))
            # Unoptimized baseline: ~28us for persistent-mode decode (batch=64, ctx=4000)
            # Optimized reduce kernel saves ~5us (18.2us -> 13us), giving ~23us total
            threshold_us = 25.0
            print(f"  Persistent MLA decode latency: {latency_us:.1f}us (threshold: <{threshold_us}us)")
            check(f"MLA decode latency < {threshold_us}us (got {latency_us:.1f}us)", latency_us < threshold_us,
                  f"Reduce kernel too slow ({latency_us:.1f}us), optimization likely not applied")
        else:
            stderr_tail = result.stderr[-500:] if result.stderr else ""
            stdout_tail = result.stdout[-300:] if result.stdout else ""
            check("MLA performance measurement", False,
                  f"Could not extract latency. stdout: {stdout_tail} stderr: {stderr_tail}")

    except Exception as e:
        check("MLA performance measurement", False, str(e)[:200])
else:
    check("MLA performance (skipped, module unavailable)", False, "MLA module not available")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
