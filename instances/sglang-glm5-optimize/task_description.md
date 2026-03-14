# GLM-5-FP8 Decode Latency Optimization

Optimize the decode latency of the GLM-5-FP8 model on 8× AMD MI355X GPUs using SGLang.

## Environment

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — on `sys.path`, used by `python3 -m sglang.*`. **Edit files HERE to modify SGLang behavior.**
- **SGLang reference checkout**: `/workspace/sglang/` — fresh `git clone` for reference only. Changes here do NOT affect the runtime.
- **Model weights**: `zai-org/GLM-5-FP8` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_glm5.sh` — runs `sglang.bench_one_batch` with fixed workload params.

## Benchmark

The benchmark takes **50-60 minutes** on first run (model loading + CUDA graph compilation).
Set `timeout: 4800` when running it. If it times out, kill all leftover sglang processes
(`ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`) before retrying.

For faster iteration during profiling, run with `--disable-cuda-graph` (skips 35-min graph
compilation, completes in ~15-20 min). The FINAL measurement must use the standard benchmark.

Read the benchmark output logs carefully to identify which backends are active (attention,
MoE, all-reduce) before optimizing. Only optimize backends that are actually in use.

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- Final metrics must use CUDA graphs (no `--disable-cuda-graph`).
- Run `bench_glm5.sh` as your LAST command.
