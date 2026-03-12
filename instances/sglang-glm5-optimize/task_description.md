# GLM-5-FP8 Decode Latency Optimization

Optimize the decode latency of the GLM-5-FP8 model on 8× AMD MI355X GPUs using SGLang.

## Environment (read carefully)

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — this is on `sys.path` and is what `python3 -m sglang.*` uses. If you need to modify SGLang source, edit files here.
- **SGLang reference checkout**: `/workspace/sglang/` — a fresh `git clone` for reference. Changes here do NOT affect the runtime.
- **Model weights**: `zai-org/GLM-5-FP8` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_glm5.sh` — pre-built and working. Run it first to establish a baseline.

## Step 1 — Establish Baseline

Run the benchmark immediately:
```bash
bash /workspace/bench_glm5.sh
```
This loads the model with TP=8, runs decode, and prints `Decode median (ms): <value> | tp=8 batch=1`.
Update `/workspace/optimization_state.json` with the baseline result.

## Step 2 — Profile and Optimize

After the baseline, focus on reducing decode latency through:
1. **Configuration tuning**: environment variables (`GPU_MAX_HW_QUEUES`, `PYTORCH_TUNABLEOP_ENABLED`, `HSA_ENABLE_SDMA`), `--mem-fraction-static`, torch.compile flags.
2. **Kernel-level profiling**: use `rpd` or `torch.profiler` to find hot kernels. Look at attention, MoE dispatch, all-reduce.
3. **Source-level changes in `/sgl-workspace/sglang/`**: optimize attention backends, fused MoE kernels, reduce synchronization, improve CUDA graph capture.

After each optimization attempt, re-run `bash /workspace/bench_glm5.sh` to measure the effect.

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use `pgrep ... | xargs kill`.
- Read error messages carefully and fix the root cause.
- Final metrics must use CUDA graphs (no `--disable-cuda-graph`).
- Run `bench_glm5.sh` as your LAST command.
- Do NOT modify the benchmark script's immutable parameters (model, tp, batch, input/output lengths).
