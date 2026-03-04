# Feature: Add nhead=8 Support for MLA Decode

## Background

AITER's Multi-head Latent Attention (MLA) decode uses an optimized ASM kernel
(`mla_decode_stage1_asm_fwd`) that only supports `nhead` values in {16, 32, 64, 128}.
Models with nhead=8 (e.g., smaller DeepSeek variants) cannot use this optimized path
and fall back to slower implementations.

## Task

Add nhead=8 support for MLA decode in `aiter/mla.py` by using a **padding strategy**:

1. When `nhead == 8` and the non-persistent (ASM) decode path is selected:
   - Pad `q` from shape `(total_seqlen, 8, qk_head_dim)` to `(total_seqlen, 16, qk_head_dim)` with zeros
   - Allocate a padded output buffer `o_padded` of shape `(total_seqlen, 16, v_head_dim)`
2. Call the ASM kernel with the padded nhead=16 tensors
3. After the kernel returns, slice the output back to the original nhead=8

This approach avoids modifying the ASM kernel itself.

## Working Directory

- AITER is cloned at `/workspace/aiter`
- The main file to modify is `aiter/mla.py`
- Tests are in `op_tests/test_mla.py`

## Implementation Hints

- Look at how `mla_decode_fwd` dispatches between persistent and non-persistent paths
- The padding should only apply to the non-persistent ASM path (`mla_decode_stage1_asm_fwd`)
- Ensure the padding and slicing happen on the same device as the input tensors
- The test should verify correctness against a torch golden reference

## Verification

```bash
cd /workspace/aiter
/opt/venv/bin/python3 /workspace/test_harness.py
```

Success: All nhead=8 test cases pass against the torch golden reference.
