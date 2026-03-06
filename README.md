# amdpilot-evals

Evaluation instances for [amdpilot](https://github.com/Arist12/amdpilot). Each instance is a real-world AMD GPU task derived from actual GitHub PRs/issues.

## Scripts

### `scripts/curate_eval.py` — PR-to-Eval Curation

Generates an eval instance from a merged GitHub PR:

```bash
python evals/scripts/curate_eval.py --pr https://github.com/sgl-project/sglang/pull/18903
python evals/scripts/curate_eval.py --pr sgl-project/sglang/18903 --generate-test
python evals/scripts/curate_eval.py --pr sgl-project/sglang/18903 --generate-test \
  --model-url http://your-server:30000/v1
```

### `scripts/run_issue.py` — Autonomous Issue Resolution

Resolves a GitHub issue end-to-end (env build + test generation + agent run):

```bash
python evals/scripts/run_issue.py https://github.com/sgl-project/sglang/issues/12345
python evals/scripts/run_issue.py https://github.com/sgl-project/sglang/issues/12345 \
  --model-url http://your-server:30000/v1
# Or via CLI:
amdpilot issue https://github.com/sgl-project/sglang/issues/12345
```

## Data Curation Pipeline

### From a Merged PR

1. Fetch PR metadata via `gh` (title, body, files, merge commit)
2. Resolve parent commit (`merge_commit~1`) — the repo state BEFORE the fix
3. Generate `task_description.md` from PR body (symptom only, no solution)
4. Generate `Dockerfile` using the ROCm base image + repo at parent commit
5. Generate `test_harness.py` (LLM-generated or manual)
6. Generate `task.yaml` with `stages: auto`

### Data Leak Prevention

- Repo is checked out at `merge_commit~1` — the fix does NOT exist
- `task_description.md` describes the symptom/bug only, never the solution code
- `test_harness.py` should test expected behavior, not specific code patterns

## Test Harness Quality

| Classification | Description | Instances |
|----------------|-------------|-----------|
| **STRONG** | Runtime checks (import, exec, GPU behavior, perf metrics) | fused-moe-fix, rotary-crash, qwen35-rope-fix, moe-align-optimize, sampling-optimize, sigmoid-fastmath, mla-reduce-optimize |
| **MEDIUM** | Mix of static + runtime, or source inspection + import | vllm-encoder-rocm, vllm-ck-mxfp4-moe |

## Base Image

All evals default to `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226`:
- `/opt/venv/bin/python3` with ROCm PyTorch, triton, aiter, composable kernel
- `/sgl-workspace/aiter` — aiter source and compiled kernels
- Never use `pip install -e .` on target repos

The base image can be customized per instance by modifying the Dockerfile. When running an eval, if the Docker image is not found locally, amdpilot will automatically build it from the instance's Dockerfile.

## Available Instances (9)

| Instance | Category | Difficulty | Harness | Source |
|----------|----------|------------|---------|--------|
| sglang-fused-moe-fix | Bug Fix | Easy | STRONG | sglang #19840 |
| sglang-rotary-crash | Bug Fix | Medium | STRONG | sglang #18903 |
| sglang-qwen35-rope-fix | Bug Fix | Medium | STRONG | sglang #18753 |
| vllm-encoder-rocm | Bug Fix | Hard | MEDIUM | vllm #35334 |
| vllm-ck-mxfp4-moe | Feature | Hard | MEDIUM | vllm #34301 |
| aiter-moe-align-optimize | Optimization | Hard | STRONG | aiter #1869 |
| aiter-sampling-optimize | Optimization | Hard | STRONG | aiter #2034 |
| aiter-sigmoid-fastmath | Optimization | Hard | STRONG | aiter #1879 |
| aiter-mla-reduce-optimize | Optimization | Very Hard | STRONG | aiter #1896 |

## Running an Eval

```bash
# Run directly (Docker image auto-builds from Dockerfile)
uv run amdpilot run evals/instances/<name>/task.yaml

# Or build the Docker image manually first
cd evals/instances/<name> && docker build -t amdpilot-eval-<name> .
uv run amdpilot run evals/instances/<name>/task.yaml

# List all instances
python evals/shared/eval_runner.py --list

# Build all Docker images
python evals/shared/eval_runner.py --all --build-only
```

## Model Endpoint

Eval instances default to `http://localhost:30000/v1`. Override with:

```bash
# Environment variable
export AMDPILOT_MODEL_URL="http://your-server:30000/v1"
uv run amdpilot run evals/instances/<name>/task.yaml

# Or CLI override
uv run amdpilot run evals/instances/<name>/task.yaml --model-url http://your-server:30000/v1
```
