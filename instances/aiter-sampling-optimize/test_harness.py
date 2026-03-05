#!/usr/bin/env python3
"""Test harness for aiter-sampling-optimize. Runtime correctness + performance."""
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
print("aiter-sampling-optimize test harness")
print("=" * 60)

import torch
from aiter.ops.sampling import top_k_top_p_sampling_from_probs
check("Import sampling ops", True)

device = torch.device("cuda:0")
check("GPU available", torch.cuda.is_available())
torch.manual_seed(42)

# Correctness
print("\n--- Correctness ---")
batch, vocab = 4, 32000
probs = torch.softmax(torch.randn(batch, vocab, device=device), dim=-1)
result = top_k_top_p_sampling_from_probs(probs, None, None, 5, None, 0.9, True)
check("Valid sample indices", torch.all(result >= 0).item() and torch.all(result < vocab).item())

# Performance
print("\n--- Performance ---")
probs_lg = torch.softmax(torch.randn(1, 128256, device=device), dim=-1)
for _ in range(20): top_k_top_p_sampling_from_probs(probs_lg, None, None, 1, None, 0.9, True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(200): top_k_top_p_sampling_from_probs(probs_lg, None, None, 1, None, 0.9, True)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) * 1000 / 200
print(f"  Avg latency (bs=1, vocab=128256, k=1): {ms:.3f}ms")
check(f"Latency < 0.35ms (got {ms:.3f}ms)", ms < 0.35)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
