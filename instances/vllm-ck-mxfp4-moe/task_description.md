# vLLM MXFP4 quantization lacks fused MoE support on ROCm

vLLM's MXFP4 quantization layer has no fused MoE support on ROCm. Models using MXFP4 quantization with MoE layers fall back to dense per-expert computation, which is significantly slower. The MXFP4 quantization module needs a fused MoE backend for AMD GPUs using AITER's CK operators.

## Affected Files

- `vllm/_aiter_ops.py`
- `vllm/model_executor/layers/quantization/mxfp4.py`

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
