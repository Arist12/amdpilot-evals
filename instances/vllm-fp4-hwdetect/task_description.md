# Bug: vLLM Crashes on MI300X Due to FP4 BMM Hardware Mismatch

## Symptom

vLLM crashes during model loading on AMD MI300X (gfx942) GPUs with:

```
RuntimeError: MXFP4 quantization is not supported on gfx942
```

This happens with default settings (`VLLM_ROCM_USE_AITER=1`) when loading
models that use FP4 quantization (e.g., DeepSeek-V3).

## Root Cause

`VLLM_ROCM_USE_AITER_FP4BMM` defaults to `True` for ALL AMD GPUs, but FP4
(MXFP4) is only supported on MI325X/MI350X (gfx950 architecture). MI300X
(gfx942), which is ~90% of deployed AMD GPUs, does not have FP4 hardware support.

The code in `vllm/_aiter_ops.py` only checks environment variables, never
querying actual hardware capabilities.

## Task

Modify `vllm/_aiter_ops.py` to:
1. Check hardware FP4 capability via AITER's `is_fp4_avail()` function
2. Auto-disable FP4 BMM on unsupported hardware (gfx942)
3. Log an informative message when falling back to FP8
4. Handle the case where AITER's arch_info module is unavailable

## Expected Behavior After Fix

- MI300X (gfx942): FP4 automatically disabled, falls back to FP8, model loads fine
- MI325X/MI350X (gfx950): FP4 continues to work as before
- No user intervention required

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
