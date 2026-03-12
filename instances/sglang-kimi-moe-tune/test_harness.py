#!/usr/bin/env python3
"""Test harness for Kimi K2.5 fused_moe_triton optimization.

Scoring is based on actual kernel speedup, not pass/fail checks.
This forces the agent to produce configs that genuinely improve performance.

  Tier 0 (15 pts): Profiling evidence — agent MUST profile before optimizing
  Tier 1 (10 pts): Config files exist with correct format
  Tier 2 (10 pts): Tuning infrastructure supports int4_w4a16
  Tier 3 (65 pts): Kernel benchmark — score proportional to speedup ratio

SCORE: 0-100 (continuous)

Tier 0 enforces the profiling-first workflow. Without profiling evidence,
Tier 3 is capped at 50% — blind config tuning cannot achieve full marks.

Achievable speedup reference (fp16 proxy benchmark, E=384):
  - Default config (pathological): BLOCK_SIZE_M=16, N=32, K=64
  - Well-tuned config: ~1.4-1.6x speedup at large batch sizes
  - Excellent config: ~1.6-1.8x speedup
"""

import json
import math
import os
import sys
import time

SGLANG_DIR = "/workspace/sglang"
CONFIG_BASE = os.path.join(
    SGLANG_DIR,
    "python/sglang/srt/layers/moe/fused_moe_triton/configs",
)
TUNING_DIR = os.path.join(SGLANG_DIR, "benchmark/kernels/fused_moe_triton")

# Kimi K2.5 model parameters
EXPECTED_E = 384
EXPECTED_N = 128
EXPECTED_DTYPE = "int4_w4a16"

# Config file search paths
TRITON_VERSION_DIRS = ["triton_3_4_0", "triton_3_5_1", "triton_3_6_0"]
UP_CONFIG_NAME = (
    f"E={EXPECTED_E},N={EXPECTED_N},device_name=,dtype={EXPECTED_DTYPE}.json"
)
DOWN_CONFIG_NAME = (
    f"E={EXPECTED_E},N={EXPECTED_N},device_name=,dtype={EXPECTED_DTYPE}_down.json"
)

# Batch sizes for config coverage check
REQUIRED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
REQUIRED_CONFIG_FIELDS = [
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
    "GROUP_SIZE_M", "num_warps", "num_stages",
]
VALID_BLOCK_SIZES = {16, 32, 64, 128, 256}
VALID_NUM_WARPS = {1, 2, 4, 8}
VALID_NUM_STAGES = {0, 1, 2, 3, 4, 5}
VALID_GROUP_SIZES = {1, 2, 4, 8, 16, 32, 64}

# Kernel benchmark parameters — use REAL Kimi K2.5 dimensions
# hidden_size=7168, shard_intermediate_size=512 (with TP=8)
# Large batch sizes where config tuning shows significant impact
BENCH_HIDDEN_SIZE = 7168
BENCH_SHARD_SIZE = 512
BENCH_TOPK = 8
BENCH_BATCH_SIZES = [512, 2048, 4096, 8192, 16384]
BENCH_WARMUP_ITERS = 5
BENCH_TIMING_ITERS = 20

# Tier 0: Profiling evidence paths
# rocprof --stats creates results.stats.csv; torch.profiler creates trace.json
PROFILING_EVIDENCE_PATHS = [
    "/workspace/results.stats.csv",
    "/workspace/sglang/results.stats.csv",
    "/workspace/results.csv",
    "/workspace/trace.json",
    "/workspace/profiling_results.txt",
    "/workspace/profile_output.txt",
]
PROFILING_EVIDENCE_GLOBS = [
    "/workspace/**/results.stats.csv",
    "/workspace/**/*prof*.csv",
    "/workspace/**/*trace*.json",
]
PROFILING_KEYWORDS_IN_STATE = [
    "rocprof", "profiler", "gpu_time", "kernel_time", "cuda_time",
    "fused_moe_kernel", "top_bottlenecks", "ProfilerActivity",
]

# Tier 3 scoring: speedup_ratio maps to points
# 1.0x -> 0 pts, 1.5x -> 42 pts, 1.8x+ -> 65 pts (cap)
TIER3_MAX_POINTS = 65.0
TIER3_SPEEDUP_FOR_MAX = 1.8  # speedup at which max points are awarded


def find_config_file(config_name):
    for vdir in TRITON_VERSION_DIRS:
        path = os.path.join(CONFIG_BASE, vdir, config_name)
        if os.path.isfile(path):
            return path
    return None


def load_config_as_runtime(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(key): val for key, val in raw.items()}


def validate_config_entry(entry):
    for field in REQUIRED_CONFIG_FIELDS:
        if field not in entry:
            return False, f"missing field '{field}'"
    checks = [
        (entry.get("BLOCK_SIZE_M", 0), VALID_BLOCK_SIZES, "BLOCK_SIZE_M"),
        (entry.get("BLOCK_SIZE_N", 0), VALID_BLOCK_SIZES, "BLOCK_SIZE_N"),
        (entry.get("BLOCK_SIZE_K", 0), VALID_BLOCK_SIZES, "BLOCK_SIZE_K"),
        (entry.get("GROUP_SIZE_M", 0), VALID_GROUP_SIZES, "GROUP_SIZE_M"),
        (entry.get("num_warps", 0), VALID_NUM_WARPS, "num_warps"),
        (entry.get("num_stages", 0), VALID_NUM_STAGES, "num_stages"),
    ]
    for val, valid_set, name in checks:
        if val not in valid_set:
            return False, f"{name}={val} not in {valid_set}"
    return True, ""


def speedup_to_points(speedup_ratio):
    """Convert geometric mean speedup to tier 3 points (0 to TIER3_MAX_POINTS)."""
    if speedup_ratio <= 1.0:
        return 0.0
    frac = (speedup_ratio - 1.0) / (TIER3_SPEEDUP_FOR_MAX - 1.0)
    return min(TIER3_MAX_POINTS, max(0.0, frac * TIER3_MAX_POINTS))


def main():
    print("=" * 60)
    print("Kimi K2.5 fused_moe_triton Optimization Test Harness v2")
    print("=" * 60)

    tier0_points = 0.0
    tier1_points = 0.0
    tier2_points = 0.0
    tier3_points = 0.0
    profiling_done = False

    # ========== TIER 0: Profiling evidence (15 points) ==========
    print("\n--- Tier 0: Profiling Evidence (15 pts) ---")
    print("  (You MUST run rocprof or torch.profiler before optimizing)")

    import glob as glob_mod

    # Check 1: profiling output files exist (8 pts)
    prof_files_found = []
    for path in PROFILING_EVIDENCE_PATHS:
        if os.path.isfile(path):
            prof_files_found.append(path)
    for pattern in PROFILING_EVIDENCE_GLOBS:
        prof_files_found.extend(glob_mod.glob(pattern, recursive=True))

    prof_files_found = list(set(prof_files_found))
    if prof_files_found:
        tier0_points += 8
        profiling_done = True
        print(f"  [PASS] Profiling output found: {prof_files_found[0]}")
        for pf in prof_files_found[1:]:
            print(f"         Also: {pf}")
    else:
        print("  [FAIL] No profiling output files found.")
        print("         Expected: rocprof --stats creates results.stats.csv")
        print("         Or: torch.profiler creates trace.json")

    # Check 2: optimization_state.json references profiling results (7 pts)
    state_path = "/workspace/optimization_state.json"
    if os.path.isfile(state_path):
        try:
            with open(state_path) as f:
                state_content = f.read().lower()
            kw_found = [kw for kw in PROFILING_KEYWORDS_IN_STATE if kw.lower() in state_content]
            if len(kw_found) >= 2:
                tier0_points += 7
                profiling_done = True
                print(f"  [PASS] optimization_state.json references profiling: {kw_found[:3]}")
            elif len(kw_found) == 1:
                tier0_points += 3
                print(f"  [PARTIAL] optimization_state.json has 1 profiling ref: {kw_found}")
            else:
                print("  [FAIL] optimization_state.json has no profiling references")
        except Exception:
            print("  [FAIL] Could not read optimization_state.json")
    else:
        print("  [FAIL] optimization_state.json not found")

    if not profiling_done:
        print()
        print("  WARNING: No profiling evidence detected.")
        print("  Tier 3 (kernel benchmark) will be CAPPED at 50%.")
        print("  Run: rocprof --stats /opt/venv/bin/python3 /workspace/test_harness.py 2>&1 | tail -40")
        print("  Then record results in optimization_state.json under profiling_summary.")

    print(f"  Tier 0 score: {tier0_points:.0f}/15")

    # ========== TIER 1: Config file existence and format (10 points) ==========
    print("\n--- Tier 1: Config File Validation (10 pts) ---")

    up_path = find_config_file(UP_CONFIG_NAME)
    down_path = find_config_file(DOWN_CONFIG_NAME)

    up_configs = None
    down_configs = None

    if up_path:
        tier1_points += 2
        print(f"  [PASS] Up config found: {up_path}")
    else:
        print(f"  [FAIL] Up config not found: {UP_CONFIG_NAME}")

    if down_path:
        tier1_points += 1
        print(f"  [PASS] Down config found: {down_path}")
    else:
        print(f"  [FAIL] Down config not found: {DOWN_CONFIG_NAME}")

    if up_path:
        try:
            up_configs = load_config_as_runtime(up_path)
            if (isinstance(up_configs, dict) and len(up_configs) > 0 and
                    all(isinstance(k, int) for k in up_configs.keys())):
                tier1_points += 2
                print(f"  [PASS] Up config: batch-size-keyed dict with {len(up_configs)} entries")
            else:
                print("  [FAIL] Up config: not a valid batch-size-keyed dict")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [FAIL] Up config load error: {e}")

    if down_path:
        try:
            down_configs = load_config_as_runtime(down_path)
            if (isinstance(down_configs, dict) and len(down_configs) > 0 and
                    all(isinstance(k, int) for k in down_configs.keys())):
                tier1_points += 1
                print(f"  [PASS] Down config: batch-size-keyed dict with {len(down_configs)} entries")
            else:
                print("  [FAIL] Down config: not a valid batch-size-keyed dict")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [FAIL] Down config load error: {e}")

    # Validate config entries
    if up_configs:
        all_valid = True
        for bs, entry in up_configs.items():
            ok, reason = validate_config_entry(entry)
            if not ok:
                all_valid = False
                print(f"  [FAIL] Up config batch={bs}: {reason}")
                break
        if all_valid:
            tier1_points += 2
            print("  [PASS] All up config entries have valid fields")

        covered = sum(1 for bs in REQUIRED_BATCH_SIZES if bs in up_configs)
        if covered >= len(REQUIRED_BATCH_SIZES) * 0.6:
            tier1_points += 2
            print(f"  [PASS] Up config covers {covered}/{len(REQUIRED_BATCH_SIZES)} batch sizes")
        else:
            print(f"  [FAIL] Up config covers only {covered}/{len(REQUIRED_BATCH_SIZES)} batch sizes")

    print(f"  Tier 1 score: {tier1_points:.0f}/10")

    # ========== TIER 2: Tuning infrastructure (10 points) ==========
    print("\n--- Tier 2: Tuning Infrastructure (10 pts) ---")

    tuning_script = os.path.join(TUNING_DIR, "tuning_fused_moe_triton.py")
    tuning_sep_script = os.path.join(TUNING_DIR, "tuning_fused_moe_triton_sep.py")
    common_utils = os.path.join(TUNING_DIR, "common_utils.py")

    for script_path, name, pts in [
        (tuning_script, "tuning_fused_moe_triton.py", 2),
        (tuning_sep_script, "tuning_fused_moe_triton_sep.py", 1),
        (common_utils, "common_utils.py", 2),
    ]:
        if os.path.isfile(script_path):
            with open(script_path) as f:
                content = f.read()
            if "int4_w4a16" in content:
                tier2_points += pts
                print(f"  [PASS] {name} references int4_w4a16")
            else:
                print(f"  [FAIL] {name} missing int4_w4a16 support")
        else:
            print(f"  [FAIL] {name} not found")

    if os.path.isfile(common_utils):
        with open(common_utils) as f:
            content = f.read()
        if "use_int4_w4a16" in content and ("// 2" in content or "//2" in content):
            tier2_points += 2
            print("  [PASS] common_utils.py adjusts N for int4 packing")
        else:
            print("  [FAIL] common_utils.py missing int4 N adjustment")

    if os.path.isfile(tuning_script):
        with open(tuning_script) as f:
            content = f.read()
        if ("uint8" in content or "torch.uint8" in content) and "int4" in content:
            tier2_points += 2
            print("  [PASS] Tuning script creates uint8 packed int4 tensors")
        else:
            print("  [FAIL] Tuning script missing uint8 packed int4 tensor creation")

    if os.path.isfile(common_utils):
        with open(common_utils) as f:
            content = f.read()
        if "text_config" in content and "get_text_config" in content:
            tier2_points += 1
            print("  [PASS] common_utils.py handles text_config for encoder-decoder")
        else:
            print("  [FAIL] common_utils.py missing text_config handling")

    print(f"  Tier 2 score: {tier2_points:.0f}/10")

    # ========== TIER 3: Kernel benchmark (80 points) ==========
    print("\n--- Tier 3: Kernel Benchmark (80 pts) ---")
    print(f"  Benchmark dims: E={EXPECTED_E}, hidden={BENCH_HIDDEN_SIZE}, shard={BENCH_SHARD_SIZE}")
    print(f"  Batch sizes: {BENCH_BATCH_SIZES}")

    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    if not gpu_available:
        print("  [SKIP] No GPU available")
        # Minimal partial credit for having valid static configs
        if up_configs and down_configs:
            tier3_points = 5.0
            print("  Awarded 5 pts for structurally valid configs (no GPU)")
        else:
            tier3_points = 0.0

    elif not up_configs:
        print("  [FAIL] No valid up config to benchmark")
        tier3_points = 0.0

    else:
        sglang_python = os.path.join(SGLANG_DIR, "python")
        if sglang_python not in sys.path:
            sys.path.insert(0, sglang_python)

        try:
            # Initialize ServerArgs (required by config lookup)
            try:
                from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
                set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", port=0))
            except Exception:
                pass

            from sglang.srt.layers.moe.fused_moe_triton import override_config
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
            from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
            from sglang.srt.layers.moe.topk import TopKConfig, select_experts

            torch.set_default_device("cuda")

            E = EXPECTED_E
            hidden_size = BENCH_HIDDEN_SIZE
            shard = BENCH_SHARD_SIZE
            topk = BENCH_TOPK

            speedups = []
            print()

            for M in BENCH_BATCH_SIZES:
                try:
                    x = torch.randn(M, hidden_size, dtype=torch.float16)
                    w1 = torch.randn(E, shard, hidden_size, dtype=torch.float16)
                    w2 = torch.randn(E, hidden_size, shard // 2, dtype=torch.float16)
                    gating = torch.randn(M, E, dtype=torch.float32)

                    topk_config = TopKConfig(top_k=topk, renormalize=True)
                    topk_output = select_experts(x, gating, topk_config)
                    moe_config = MoeRunnerConfig(inplace=True)

                    def run_kernel(cfg=None):
                        ctx = override_config(cfg) if cfg else override_config(None)
                        with ctx:
                            fused_moe(x, w1, w2, topk_output, moe_runner_config=moe_config)

                    # Warmup
                    for _ in range(BENCH_WARMUP_ITERS):
                        run_kernel(None)
                    torch.cuda.synchronize()

                    # Benchmark default config
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(BENCH_TIMING_ITERS):
                        run_kernel(None)
                    torch.cuda.synchronize()
                    default_us = (time.perf_counter() - t0) * 1e6 / BENCH_TIMING_ITERS

                    # Benchmark with loaded config
                    closest_bs = min(up_configs.keys(), key=lambda k: abs(k - M))
                    loaded_cfg = up_configs[closest_bs]

                    # Warmup tuned config
                    for _ in range(BENCH_WARMUP_ITERS):
                        run_kernel(loaded_cfg)
                    torch.cuda.synchronize()

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(BENCH_TIMING_ITERS):
                        run_kernel(loaded_cfg)
                    torch.cuda.synchronize()
                    tuned_us = (time.perf_counter() - t0) * 1e6 / BENCH_TIMING_ITERS

                    ratio = default_us / tuned_us
                    speedups.append(ratio)
                    tokens_per_expert = M * topk / E
                    print(f"  M={M:>5}: default={default_us:>10.1f}us, tuned={tuned_us:>10.1f}us, "
                          f"speedup={ratio:.3f}x (tpe={tokens_per_expert:.0f})")

                    # Clean up GPU memory
                    del x, w1, w2, gating, topk_output
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  M={M:>5}: ERROR — {e}")
                    torch.cuda.empty_cache()

            print()

            if speedups:
                # Geometric mean of speedup ratios
                geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
                tier3_points = speedup_to_points(geo_mean)
                print(f"  Geometric mean speedup: {geo_mean:.3f}x")
                print(f"  Tier 3 score: {tier3_points:.1f}/{TIER3_MAX_POINTS:.0f}")
                print(f"  (1.0x=0pts, 1.5x={speedup_to_points(1.5):.0f}pts, "
                      f"1.8x={speedup_to_points(1.8):.0f}pts)")
            else:
                print("  [FAIL] All batch sizes failed")
                tier3_points = 0.0

        except Exception as e:
            print(f"  [ERROR] Kernel benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            # Minimal partial credit for valid static configs
            if up_configs and down_configs:
                tier3_points = 5.0
                print("  Awarded 5 pts for structurally valid configs (benchmark crashed)")
            else:
                tier3_points = 0.0

    # ========== Profiling gate on Tier 3 ==========
    tier3_raw = tier3_points
    if not profiling_done and tier3_points > TIER3_MAX_POINTS * 0.5:
        tier3_points = TIER3_MAX_POINTS * 0.5
        print(f"\n  ** Tier 3 CAPPED: {tier3_raw:.1f} -> {tier3_points:.1f} (no profiling evidence) **")
        print(f"  ** Run rocprof first to unlock full Tier 3 score **")

    # ========== Summary ==========
    total_score = tier0_points + tier1_points + tier2_points + tier3_points
    print()
    print("=" * 60)
    print(f"  Tier 0 (Profiling):         {tier0_points:>5.1f} / 15")
    print(f"  Tier 1 (Config files):      {tier1_points:>5.1f} / 10")
    print(f"  Tier 2 (Infrastructure):    {tier2_points:>5.1f} / 10")
    print(f"  Tier 3 (Kernel benchmark):  {tier3_points:>5.1f} / 65")
    if not profiling_done and tier3_raw > tier3_points:
        print(f"         (uncapped: {tier3_raw:.1f}, capped due to missing profiling)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:                      {total_score:>5.1f} / 100")
    print()
    print(f"SCORE: {total_score:.1f}")
    sys.exit(0 if total_score >= 20.0 else 1)


if __name__ == "__main__":
    main()
