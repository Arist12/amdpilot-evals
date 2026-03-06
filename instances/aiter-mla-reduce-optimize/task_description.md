# Optimize MLA Reduce Kernel

The Multi-Latent Attention (MLA) reduce kernel in AITER (`csrc/kernels/mla/reduce.cu`) has high latency that impacts end-to-end inference performance for models using MLA (e.g., DeepSeek variants).

Profiling with a sparse MLA workload (`batch=1, context=4000, nhead=16,2, dtype=bf16, kvd=bf16`) shows the reduce kernel takes approximately 18.2 microseconds. The target is to reduce this to under 13 microseconds (at least 25% improvement).

The kernel performs reduction of partial attention outputs across splits. Key bottlenecks include:
- Suboptimal handling of different split counts (small vs. large)
- Memory access patterns in the reduce phase
- Synchronization overhead between compute and memory operations

The kernel source is at: `/sgl-workspace/aiter/csrc/kernels/mla/reduce.cu`

After making changes, rebuild aiter:
```bash
cd /sgl-workspace/aiter && /opt/venv/bin/python3 setup.py develop
```

## Environment

- AITER at `/sgl-workspace/aiter`
- Use `/opt/venv/bin/python3`
- AMD MI355X GPU (gfx950)

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
