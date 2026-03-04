# amdpilot-evals

Evaluation instances for the [amdpilot](https://github.com/Arist12/amdpilot) agentic system. Each instance represents a real-world AMD GPU workload derived from actual issues and PRs in production open-source projects.

## Instance Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Bug Fix** | Fix AMD/ROCm-specific bugs | Missing import guards, hardware detection |
| **Feature** | Implement AMD-specific functionality | New kernels, operator support |
| **Performance** | Optimize for AMD GPUs | Kernel tuning, tile-size selection |

## Instance Structure

Each instance under `instances/` contains:

```
instances/<name>/
├── task.yaml              # amdpilot job config
├── task_description.md    # Human-readable task description (injected into agent prompt)
├── Dockerfile             # Docker environment setup
├── test_harness.py        # Verification script (exit 0 = pass, exit 1 = fail)
├── metadata.json          # Category, difficulty, source PR, expected LOC
└── setup.sh               # (optional) Additional setup commands
```

## Running an Eval

```bash
# 1. Build the Docker image for the instance
cd instances/<name>
docker build -t amdpilot-eval-<name> .

# 2. Run with amdpilot
cd /path/to/amdpilot
uv run amdpilot run /path/to/amdpilot-evals/instances/<name>/task.yaml
```

## Available Instances

| Instance | Category | Source | Difficulty | LOC |
|----------|----------|--------|------------|-----|
| `sglang-fused-moe-fix` | Bug Fix | [sglang PR #19840](https://github.com/sgl-project/sglang/pull/19840) | Easy | ~3 |
| `aiter-mla-nhead8` | Feature | [aiter PR #2138](https://github.com/ROCm/aiter/pull/2138) | Medium | ~36 |
| `vllm-fp4-hwdetect` | Bug Fix | [vllm PR #34647](https://github.com/vllm-project/vllm/pull/34647) | Medium | ~39 |

## Design Principles

1. **Derived from real PRs** — every instance is based on an actual merged or open PR
2. **Programmatic verification** — `test_harness.py` gives a binary pass/fail, no human judgment
3. **Self-contained Docker env** — each instance ships its own Dockerfile
4. **Agent-agnostic** — instances work with amdpilot but the task/test are decoupled
