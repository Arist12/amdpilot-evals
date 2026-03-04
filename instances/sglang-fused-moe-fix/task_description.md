# Bug: NameError in fused_moe.py on ROCm

## Symptom

All MoE models (Kimi-K2.5, Mixtral-8x7B-FP8, gpt-oss) crash on AMD ROCm GPUs with:

```
NameError: name 'get_global_server_args' is not defined
```

The error occurs in `fused_experts_impl()` inside `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`.

## Context

A recent commit (`ee5ccde`) added a fused MoE sum-all-reduce optimization. The `get_global_server_args` function was imported conditionally (only for CUDA) but used unconditionally in `fused_experts_impl`. On ROCm/HIP, the import never executes so the function is undefined at call time.

This blocks ALL MoE model inference on AMD GPUs.

## Environment

- SGLang cloned at `/workspace/sglang` (commit `ee5ccde`)
- ROCm GPU available
- Python environment at `/opt/venv/bin/python3`

## Expected Fix

The fix should:
1. Prevent `get_global_server_args()` from being called on non-CUDA platforms
2. Not change behavior on CUDA (the optimization should still work there)
3. Preserve the existing code structure

## Verification

Run the test harness:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The test checks that the module can be loaded on ROCm without NameError and that the CUDA-specific code path is properly guarded.
