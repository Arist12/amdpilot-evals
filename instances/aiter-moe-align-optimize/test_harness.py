#!/usr/bin/env python3
"""Test harness for aiter-moe-align-optimize. Runtime correctness + performance."""
import sys, time
sys.path.insert(0, "/sgl-workspace/aiter")

checks_passed = 0
checks_total = 0

def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition: checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition: msg += f": {detail}"
    print(msg)

print("=" * 60)
print("aiter-moe-align-optimize test harness")
print("=" * 60)

import torch, aiter
check("Import aiter", True)

device = torch.device("cuda:0")
check("GPU available", torch.cuda.is_available())

# Correctness
print("\n--- Correctness ---")
num_tokens, num_experts, topk, block_size = 128, 8, 2, 128
total = num_tokens * topk
max_out = num_experts * (total // num_experts + block_size)

topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)
sorted_token_ids = torch.empty(max_out, dtype=torch.int32, device=device)
experts_ids = torch.empty(num_experts, dtype=torch.int32, device=device)
token_nums = torch.empty(num_experts, dtype=torch.int32, device=device)
num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)

aiter.moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad)
padded = num_tokens_post_pad.item()
check(f"Post-padded count valid ({padded} >= {total})", padded >= total)
check("Token nums sum matches", token_nums.sum().item() == total)

# Performance
print("\n--- Performance ---")
nt, ne, tk, bs = 4096, 64, 8, 128
tot = nt * tk
mo = ne * (tot // ne + bs)
ti = torch.randint(0, ne, (nt, tk), device=device, dtype=torch.int32)
st = torch.empty(mo, dtype=torch.int32, device=device)
ei = torch.empty(ne, dtype=torch.int32, device=device)
tn = torch.empty(ne, dtype=torch.int32, device=device)
ntp = torch.empty(1, dtype=torch.int32, device=device)

for _ in range(20): aiter.moe_align_block_size(ti, ne, bs, st, ei, tn, ntp)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(500): aiter.moe_align_block_size(ti, ne, bs, st, ei, tn, ntp)
torch.cuda.synchronize()
us = (time.perf_counter() - t0) * 1e6 / 500
print(f"  Avg latency (tokens=4096, E=64, topk=8): {us:.1f}us")
check(f"Latency < 140us (got {us:.1f}us)", us < 140.0)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
