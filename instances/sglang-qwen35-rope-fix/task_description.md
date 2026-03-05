# Fix Qwen3.5 Startup Failures on ROCm

## Symptom

Two separate startup failures when serving Qwen 3.5 MoE on AMD ROCm:

### Bug 1: RoPE config parsing failure
Some Qwen3.5 checkpoints provide RoPE settings in `text_config.rope_parameters`. The current `Qwen3_5MoeTextConfig.__init__` loses these values, causing:
```
ValueError: Unknown RoPE scaling type
```

### Bug 2: CuTe-DSL import failure on ROCm
`hybrid_linear_attn_backend.py` unconditionally imports CuTe DSL kernel code which depends on `cuda.bindings` (CUDA-specific, unavailable on ROCm):
```
ModuleNotFoundError: No module named 'cuda'
```

## Affected Files

- `python/sglang/srt/configs/qwen3_5.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

## Environment

- Repository: sgl-project/sglang
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
