# Qwen3.5 Fails to Start on ROCm

Serving Qwen 3.5 MoE on AMD ROCm fails at startup with two separate errors:

1. `ValueError: Unknown RoPE scaling type` -- appears during model config initialization

2. `ModuleNotFoundError: No module named 'cuda'` -- appears during attention backend import

Both errors prevent Qwen3.5 from running on AMD GPUs.

## Affected Files

- `python/sglang/srt/configs/qwen3_5.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
