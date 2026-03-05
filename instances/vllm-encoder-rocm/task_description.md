# vLLM ROCm attention backends fail on encoder self-attention

vLLM's ROCm attention backends raise NotImplementedError when handling encoder self-attention. This blocks encoder-decoder models (e.g., Whisper) from running on AMD GPUs. The error occurs in the RocmAttention and RocmAiterUnifiedAttention backend classes.

## Affected Files

- `vllm/v1/attention/backends/rocm_attn.py`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
