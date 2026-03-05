# Fix RotaryEmbedding Crash on AMD/ROCm

## Symptom

ALL models crash on ROCm with errors related to `tvm_ffi` JIT compilation or missing `nvidia-smi`/`CUDA_HOME`. The error occurs in the rotary embedding layer during model forward pass.

A recent change (PR #17934) changed the HIP rotary embedding fallback from `sgl_kernel.rotary_embedding` to a path that uses `tvm_ffi` JIT compilation requiring NVIDIA tools, which are unavailable on AMD GPUs. Since `_is_cuda=False` on HIP, the code always routes through this broken JIT fallback path.

## Affected Files

- `python/sglang/srt/layers/rotary_embedding.py`

## Key Observations

- The `RotaryEmbedding` class has subclasses with different `forward_native()` signatures (e.g., `MRotaryEmbedding`, `DeepseekScalingRotaryEmbedding`)
- The HIP path needs to bypass the NVIDIA-specific JIT path and use pure PyTorch (`forward_native()`) instead
- The fix must work for ALL RotaryEmbedding subclasses

## Environment

- Repository: sgl-project/sglang
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
