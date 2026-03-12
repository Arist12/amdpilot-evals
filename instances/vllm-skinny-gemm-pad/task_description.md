# Skinny GEMM Kernel: Add Padding Support for Non-Contiguous Tensors

## Context

vLLM's ROCm backend includes a hand-tuned "skinny GEMM" kernel (`wvSplitK`) in
`csrc/rocm/skinny_gemms.cu` optimized for matrix multiplications where one
dimension is small (N <= 4). This kernel is dispatched by the routing logic in
`vllm/model_executor/layers/utils.py` via the `rocm_unquantized_gemm_impl`
function.

Some models (e.g., `falcon-mamba-tiny-dev` and others with non-power-of-2
hidden dimensions) produce **padded tensors** — tensors whose memory stride is
larger than the logical dimension K. Currently the routing function guards
against these with an `x.is_contiguous()` check, falling back to the much
slower `torch.nn.functional.linear` for any non-contiguous input.

## Problem

1. The `wvSplitK` kernel assumes `stride == K` for all memory accesses. When
   called with a padded tensor (`stride > K`), it reads data from wrong offsets,
   producing incorrect results or crashing.

2. The `is_contiguous()` guard in `utils.py` prevents padded tensors from
   reaching the optimized kernel entirely, causing a silent performance
   regression for models that produce padded activations.

## Goal

Modify the skinny GEMM kernel to correctly handle non-contiguous (padded)
tensors so the `is_contiguous()` guard can be removed and these models benefit
from the optimized kernel path.

## Key Files

- **`csrc/rocm/skinny_gemms.cu`** — The CUDA/HIP kernel implementations.
  Contains multiple kernel variants: `wvSplitK_hf_sml_`, `wvSplitK_hf_`, and
  the dispatch macros that select which variant to launch. The kernel functions
  currently take `K` and `M` as dimension parameters but not stride parameters.

- **`vllm/model_executor/layers/utils.py`** — The `rocm_unquantized_gemm_impl`
  function that routes GEMMs to the skinny kernel. Look at the `use_skinny`
  condition around line 190 — the `is_contiguous()` check here prevents padded
  tensors from using the optimized path.

- **`tests/kernels/quantization/test_rocm_skinny_gemms.py`** — The existing test
  suite. Note the `pad_fp8()` helper function that creates padded tensors. The
  existing `test_rocm_wvsplitk_kernel` tests only test contiguous tensors.

## Required Approach

**You MUST follow this workflow:**

1. **Profile first**: Run `rocprof --stats` on the existing test to understand
   which GPU kernels are involved and how the skinny GEMM dispatch works:
   ```bash
   cd /workspace/vllm
   rocprof --stats python -c "
   import torch
   import vllm._custom_ops as ops
   from vllm.utils.platform_utils import num_compute_units
   A = torch.randn(4, 4096, dtype=torch.bfloat16, device='cuda')
   B = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
   cu = num_compute_units()
   for _ in range(100):
       ops.wvSplitK(B, A, cu)
   torch.cuda.synchronize()
   " 2>&1 | tail -40
   ```

2. **Diagnose the padding issue**: Create a padded tensor and observe the
   incorrect results:
   ```python
   import torch, torch.nn.functional as F
   import vllm._custom_ops as ops
   from vllm.utils.platform_utils import num_compute_units

   A = torch.randn(4, 4096, dtype=torch.bfloat16, device='cuda')
   B = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
   # Pad to make non-contiguous
   A_padded = F.pad(A, (0, 128), "constant", 0)[..., :-128]
   print(f"A contiguous: {A.is_contiguous()}, A_padded contiguous: {A_padded.is_contiguous()}")
   print(f"A stride: {A.stride()}, A_padded stride: {A_padded.stride()}")

   ref = torch.nn.functional.linear(A_padded, B)
   out = ops.wvSplitK(B, A_padded.reshape(-1, A_padded.size(-1)), num_compute_units())
   print(f"Max error: {(out - ref).abs().max().item()}")
   ```

3. **Fix the kernel**: Modify `csrc/rocm/skinny_gemms.cu` to accept and use
   tensor strides instead of assuming `stride == K`. You need to update:
   - Kernel function signatures to accept stride parameters
   - All memory access patterns that use K for addressing
   - The dispatch macros that call the kernels
   - Bounds checking in the write-back section

4. **Rebuild after each change**:
   ```bash
   cd /workspace/vllm && pip install -e . --no-build-isolation
   ```

5. **Remove the guard**: Once the kernel handles strides correctly, remove the
   `is_contiguous()` check from `utils.py` so padded tensors use the optimized
   path.

## Environment

- Repository: `/workspace/vllm`
- Use `python` or `/opt/venv/bin/python3` for running scripts
- ROCm GPU available with `rocprof` for profiling
- After modifying C++ files, rebuild with:
  `cd /workspace/vllm && pip install -e . --no-build-isolation`

## Verification

Run the test harness after applying your fix:
```bash
python /workspace/test_harness.py
```
