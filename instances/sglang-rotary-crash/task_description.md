# RotaryEmbedding fails on ROCm with CUDA_HOME error

ROCm CI fails during runtime initialization. On HIP, `RotaryEmbedding` dispatches through `MultiPlatformOp` and hits a fallback path that JIT-compiles with `cuda_files`. In ROCm-only environments, this triggers CUDA toolchain discovery and fails with:

```
RuntimeError: Could not find CUDA installation. Please set CUDA_HOME environment variable.
```

This breaks ALL models on ROCm since every model uses rotary embeddings.

Failing CI links:
- 2-GPU accuracy
- 1-GPU unit tests
- VLM accuracy and performance

## Affected File

- `python/sglang/srt/layers/rotary_embedding.py`

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
