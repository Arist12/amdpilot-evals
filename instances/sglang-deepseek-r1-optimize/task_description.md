# Optimize DeepSeek R1 on MI300X

## Problem

The DeepSeek-V3/R1 model on AMD MI300X (gfx942) does not use several optimized code paths because they are gated behind `_use_aiter_gfx95` (gfx950-only). This leaves significant performance on the table:

1. The fused RoPE + concatenation + KV cache write Triton kernel (`fused_qk_rope_cat_and_cache_mla`) is not used on gfx942, falling back to separate `rotary_emb` + `torch.cat` + cache write operations
2. FP8 GEMM configurations are not tuned for DeepSeek-specific matrix shapes on gfx942

Additionally, there may be a correctness bug where RoPE is applied twice in the MLA decode path (once in the model forward, once in the attention backend), which destroys accuracy.

## Expected Fix

- Broaden the fused kernel paths from gfx950-only to all HIP platforms
- Ensure RoPE is applied exactly once (not double-applied)
- Add or enable FP8 GEMM configs for DeepSeek shapes on gfx942

## Affected Files

- `python/sglang/srt/layers/attention/aiter_backend.py`
- `python/sglang/srt/layers/quantization/fp8_utils.py`
- `python/sglang/srt/models/deepseek_v2.py`

## Environment

- Repository: sgl-project/sglang
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
