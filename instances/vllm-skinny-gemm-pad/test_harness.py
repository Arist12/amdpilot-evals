#!/usr/bin/env python3
"""Test harness for vllm-skinny-gemm-pad.

Scores the agent's work on adding padding support to the wvSplitK skinny GEMM
kernel. Four scoring tiers:

  Tier 0 - Profiling Evidence       (15 pts)
  Tier 1 - Padded Tensor Tests      (40 pts)
  Tier 2 - Non-Padded Regression    (20 pts)
  Tier 3 - Integration Checks       (25 pts)
                            Total = 100 pts
"""

import glob
import math
import os
import sys
import traceback
from unittest.mock import patch

sys.path.insert(0, "/workspace/vllm")

tier_scores: dict[str, float] = {}

KERNEL_PATH = "/workspace/vllm/csrc/rocm/skinny_gemms.cu"
UTILS_PATH = "/workspace/vllm/vllm/model_executor/layers/utils.py"


def pad_tensor(weight):
    """Create a padded (non-contiguous) view — same technique as pad_fp8."""
    import torch.nn.functional as F
    num_pad = 256 // weight.element_size()
    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


def _validate_rocprof_csv(path):
    """Return True only if the CSV looks like genuine rocprof output."""
    try:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            return False
        header = lines[0].lower()
        if "name" not in header and "kernel" not in header:
            return False
        for line in lines[1:]:
            cols = line.split(",")
            if len(cols) < 3:
                return False
            has_any_large_number = False
            for col in cols:
                col = col.strip()
                if col.isdigit() and int(col) > 999:
                    has_any_large_number = True
            ns_vals = [c.strip() for c in cols if c.strip().isdigit() and int(c.strip()) > 10000]
            if not ns_vals:
                return False
        return True
    except Exception:
        return False


def test_routing_uses_skinny_kernel(n, k, m, dtype, bias_mode, padded_a, padded_b):
    """Test that rocm_unquantized_gemm_impl routes to wvSplitK for padded tensors.

    Returns (used_skinny, correct_result, detail).
    """
    import torch
    import vllm._custom_ops as ops

    torch.manual_seed(42)

    xavier = math.sqrt(2 / k)
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    BIAS = None
    if bias_mode == 1:
        BIAS = torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    elif bias_mode == 2:
        BIAS = torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1

    if padded_a:
        A = pad_tensor(A)
    if padded_b:
        B = pad_tensor(B)

    ref = torch.nn.functional.linear(A, B, BIAS)

    skinny_called = [False]
    original_wvSplitK = ops.wvSplitK

    def tracking_wvSplitK(*args, **kwargs):
        skinny_called[0] = True
        return original_wvSplitK(*args, **kwargs)

    try:
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
        with patch.object(ops, "wvSplitK", side_effect=tracking_wvSplitK):
            out = rocm_unquantized_gemm_impl(A, B, BIAS)
    except Exception as e:
        return False, False, f"routing error: {e}"

    correct = torch.allclose(out, ref, atol=1e-3, rtol=1e-2)
    max_err = (out - ref).abs().max().item()

    return skinny_called[0], correct, f"skinny={skinny_called[0]}, max_err={max_err:.6f}"


def test_direct_wvsplitk(n, k, m, dtype, bias_mode, padded_a, padded_b, xnorm=True):
    """Direct wvSplitK correctness test (bypasses routing)."""
    import torch
    import vllm._custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    torch.manual_seed(42)
    cu_count = num_compute_units()

    xavier = math.sqrt(2 / k) if xnorm else 1.0
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    BIAS = None
    if bias_mode == 1:
        BIAS = torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    elif bias_mode == 2:
        BIAS = torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1

    if padded_a:
        A = pad_tensor(A)
    if padded_b:
        B = pad_tensor(B)

    ref = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK(B, A.reshape(-1, A.size(-1)), cu_count, BIAS)

    atol = 1e-3
    rtol = 1e-2 if not xnorm else 1e-8
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    max_err = (out - ref).abs().max().item()
    return ok, f"max_err={max_err:.6f}"


def test_reduce_counting_guard(dtype):
    """Guardrail: padded tensors should NOT route through wvSplitKrc.

    The human patch did not change the wvSplitKrc / reduce-counting path.
    We therefore verify that padded tensors still avoid this kernel and get
    correct results via fallback behavior.
    """
    import torch
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return True, "non-gfx950 platform: reduce-counting path inactive"

    torch.manual_seed(42)
    n, k, m = 13, 2880, 128
    xavier = math.sqrt(2 / k)
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    A = pad_tensor(A)
    ref = torch.nn.functional.linear(A, B)

    rc_called = [False]
    original = ops.wvSplitKrc

    def tracking_wvSplitKrc(*args, **kwargs):
        rc_called[0] = True
        return original(*args, **kwargs)

    with patch.object(ops, "wvSplitKrc", side_effect=tracking_wvSplitKrc):
        out = rocm_unquantized_gemm_impl(A, B, None)

    correct = torch.allclose(out, ref, atol=1e-3, rtol=1e-2)
    max_err = (out - ref).abs().max().item()
    return (not rc_called[0]) and correct, (
        f"wvSplitKrc_called={rc_called[0]}, max_err={max_err:.6f}"
    )


# ---------------------------------------------------------------------------
# Tier 0 — Profiling Evidence (15 pts)
# ---------------------------------------------------------------------------

def score_tier0():
    pts = 0.0
    print("=" * 60)
    print("Tier 0: Profiling Evidence (15 pts)")
    print("=" * 60)

    profiling_csvs = [
        "/workspace/**/results.stats.csv",
        "/workspace/**/results.stats.txt",
        "/workspace/**/results.hip_stats.csv",
        "/tmp/**/results.stats.csv",
        "/tmp/**/results.stats.txt",
    ]

    found_valid = False
    for pattern in profiling_csvs:
        for match in glob.glob(pattern, recursive=True):
            if _validate_rocprof_csv(match):
                found_valid = True
                print(f"  [PASS] Validated rocprof stats: {match}")
                pts += 10.0
                break
            else:
                print(f"  [WARN] Found {match} but content is not valid rocprof output")
        if found_valid:
            break

    if not found_valid:
        profiling_other = [
            "/workspace/**/rocprof_*",
            "/workspace/**/*.prof",
            "/tmp/**/rocprof_*",
        ]
        for pattern in profiling_other:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                found_valid = True
                pts += 5.0
                print(f"  [PARTIAL] Profiling artifact found (unvalidated): {matches[0]}")
                break

    if not found_valid:
        for hf in glob.glob("/root/.bash_history") + glob.glob("/home/*/.bash_history"):
            try:
                with open(hf) as f:
                    if "rocprof" in f.read():
                        pts += 3.0
                        found_valid = True
                        print("  [PARTIAL] rocprof found in bash history (+3)")
                        break
            except Exception:
                pass

    if not found_valid:
        print("  [FAIL] No rocprof stats or profiling artifacts found")

    opt_patterns = ["/workspace/**/optimization_state.json"]
    found_opt = False
    for pattern in opt_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            found_opt = True
            pts += 5.0
            print(f"  [PASS] Optimization state: {matches[0]}")
            break
    if not found_opt:
        print("  [FAIL] No optimization_state.json found")

    print(f"  Tier 0 subtotal: {pts:.1f}/15.0")
    tier_scores["tier0"] = pts


# ---------------------------------------------------------------------------
# Tier 1 — Padded Tensor Correctness (40 pts)
# ---------------------------------------------------------------------------

def score_tier1():
    import torch
    pts = 0.0
    print()
    print("=" * 60)
    print("Tier 1: Padded Tensor Correctness (40 pts)")
    print("=" * 60)
    print("  Testing that padded tensors use skinny GEMM and produce correct results")
    print()

    test_configs = [
        (2, 256, 256),
        (4, 4096, 4096),
    ]
    bias_modes = [0, 1, 2]
    padding_combos = [
        (True, False),
        (False, True),
        (True, True),
    ]

    total_tests = len(test_configs) * len(bias_modes) * len(padding_combos)
    pts_per_test = 36.0 / total_tests
    passed = 0
    failed = 0

    for n, k, m in test_configs:
        for bias_mode in bias_modes:
            for padded_a, padded_b in padding_combos:
                label = (
                    f"N={n},K={k},M={m},bias={bias_mode},"
                    f"pad_a={padded_a},pad_b={padded_b}"
                )
                try:
                    used_skinny, correct, detail = test_routing_uses_skinny_kernel(
                        n, k, m, torch.bfloat16, bias_mode, padded_a, padded_b,
                    )
                    if used_skinny and correct:
                        pts += pts_per_test
                        passed += 1
                        print(f"  [PASS] {label} ({detail})")
                    elif correct and not used_skinny:
                        pts += pts_per_test * 0.3
                        failed += 1
                        print(f"  [PARTIAL] {label}: correct via F.linear fallback, not skinny kernel")
                    else:
                        failed += 1
                        print(f"  [FAIL] {label}: {detail}")
                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {label}: {e}")

    bonus_tests = [
        (4, 4096, 4096 + 1, True, True),
        (4, 4096 + 16, 4096, True, False),
    ]
    for n, k, m, pa, pb in bonus_tests:
        label = f"BONUS N={n},K={k},M={m},pad_a={pa},pad_b={pb}"
        try:
            ok, detail = test_direct_wvsplitk(n, k, m, torch.bfloat16, 0, pa, pb, xnorm=False)
            if ok:
                pts += 2.0
                passed += 1
                print(f"  [PASS] {label}")
            else:
                failed += 1
                print(f"  [FAIL] {label}: {detail}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {label}: {e}")

    pts = min(pts, 40.0)
    print(f"  Tier 1 subtotal: {pts:.1f}/40.0  ({passed} passed, {failed} failed)")
    tier_scores["tier1"] = pts


# ---------------------------------------------------------------------------
# Tier 2 — Non-Padded Regression (20 pts)
# ---------------------------------------------------------------------------

def score_tier2():
    import torch
    pts = 0.0
    print()
    print("=" * 60)
    print("Tier 2: Non-Padded Regression (20 pts)")
    print("=" * 60)

    regression_configs = [
        (1, 64, 64),
        (2, 256, 256),
        (3, 1024, 1024),
        (4, 4096, 4096),
        (1, 9216, 512),
        (4, 16384, 8192),
        (1, 64, 8),
        (4, 256, 8),
    ]

    total_tests = len(regression_configs)
    pts_per_test = 18.0 / total_tests
    passed = 0
    failed = 0

    for n, k, m in regression_configs:
        label = f"N={n},K={k},M={m} (contiguous)"
        try:
            ok, detail = test_direct_wvsplitk(n, k, m, torch.bfloat16, 0, False, False)
            if ok:
                pts += pts_per_test
                passed += 1
                print(f"  [PASS] {label}")
            else:
                failed += 1
                print(f"  [FAIL] {label}: {detail}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {label}: {e}")

    try:
        ok, detail = test_reduce_counting_guard(torch.bfloat16)
        if ok:
            pts += 2.0
            passed += 1
            print(f"  [PASS] reduce-counting padded guard ({detail})")
        else:
            failed += 1
            print(f"  [FAIL] reduce-counting padded guard: {detail}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] reduce-counting padded guard: {e}")

    print(f"  Tier 2 subtotal: {pts:.1f}/20.0  ({passed} passed, {failed} failed)")
    tier_scores["tier2"] = pts


# ---------------------------------------------------------------------------
# Tier 3 — Integration Checks (25 pts)
# ---------------------------------------------------------------------------

def score_tier3():
    pts = 0.0
    print()
    print("=" * 60)
    print("Tier 3: Integration Checks (25 pts)")
    print("=" * 60)

    # Check 1: is_contiguous() removed from use_skinny guard (8 pts)
    try:
        with open(UTILS_PATH) as f:
            src = f.read()
        in_use_skinny = False
        contiguous_in_skinny = False
        for line in src.split("\n"):
            if "use_skinny" in line and "=" in line and "reduce_counting" not in line:
                in_use_skinny = True
            if in_use_skinny and "is_contiguous" in line:
                contiguous_in_skinny = True
                break
            if in_use_skinny and line.strip() == ")":
                in_use_skinny = False

        if not contiguous_in_skinny:
            pts += 8.0
            print("  [PASS] is_contiguous() removed from use_skinny guard")
        else:
            print("  [FAIL] is_contiguous() still present in use_skinny guard")
    except Exception as e:
        print(f"  [FAIL] Could not check utils.py: {e}")

    # Check 2: Kernel source has stride parameters for wvSplitK (8 pts)
    try:
        with open(KERNEL_PATH) as f:
            ksrc = f.read()
        stride_names = ["Kbp", "Kap", "stride_a", "stride_b", "strideA", "strideB",
                        "lda", "ldb", "stride_k"]
        found_in_wvsplitk = False
        in_wvsplitk = False
        for line in ksrc.split("\n"):
            if "wvSplitK_hf_sml_" in line or "wvSplitK_hf_" in line:
                in_wvsplitk = True
            if in_wvsplitk:
                if any(name in line for name in stride_names):
                    found_in_wvsplitk = True
                    break
                if line.strip().startswith("{"):
                    in_wvsplitk = False
        if found_in_wvsplitk:
            pts += 8.0
            print("  [PASS] Kernel source contains stride parameters for wvSplitK")
        else:
            print("  [FAIL] No stride parameters in wvSplitK kernel signature")
    except Exception as e:
        print(f"  [FAIL] Could not check kernel: {e}")

    # Check 3: Kernel was rebuilt after source modification (5 pts)
    try:
        so_path = None
        for candidate in [
            "/workspace/vllm/vllm/_rocm_C.abi3.so",
            "/workspace/vllm/build/lib.linux-x86_64-cpython-312/vllm/_rocm_C.abi3.so",
        ]:
            if os.path.exists(candidate):
                so_path = candidate
                break
        if so_path is None:
            import vllm._custom_ops
            so_dir = os.path.dirname(vllm._custom_ops.__file__)
            candidate = os.path.join(so_dir, "_rocm_C.abi3.so")
            if os.path.exists(candidate):
                so_path = candidate

        if so_path:
            so_mtime = os.path.getmtime(so_path)
            cu_mtime = os.path.getmtime(KERNEL_PATH)
            if so_mtime >= cu_mtime:
                pts += 5.0
                print(f"  [PASS] _rocm_C.so rebuilt after kernel source modification")
            else:
                print(f"  [FAIL] _rocm_C.so is OLDER than kernel source — agent did not rebuild")
        else:
            print("  [FAIL] _rocm_C.abi3.so not found — extension not built")
    except Exception as e:
        print(f"  [FAIL] Could not check rebuild: {e}")

    # Check 4: Bounds checking in write section (4 pts)
    try:
        with open(KERNEL_PATH) as f:
            ksrc = f.read()
        has_bounds = "commitColumn" in ksrc or ("m + y" in ksrc and "M" in ksrc)
        if has_bounds:
            pts += 4.0
            print("  [PASS] Bounds checking present in kernel write section")
        else:
            print("  [FAIL] Missing bounds checking in kernel write section")
    except Exception as e:
        print(f"  [FAIL] Could not check bounds: {e}")

    print(f"  Tier 3 subtotal: {pts:.1f}/25.0")
    tier_scores["tier3"] = pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("vllm-skinny-gemm-pad test harness")
    print("=" * 60)
    print()

    has_ops = False
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, runtime tests will be skipped")
    except ImportError:
        print("ERROR: torch not importable")

    try:
        import vllm._custom_ops as ops  # noqa: F401
        print("vllm._custom_ops imported successfully")
        has_ops = True
    except Exception as e:
        print(f"WARNING: vllm._custom_ops import failed: {e}")
        print("Runtime tests (Tier 1, Tier 2) will score 0.")
        print("You must build vllm first: cd /workspace/vllm && "
              "VLLM_TARGET_DEVICE=rocm pip install -e . --no-build-isolation")

    print()

    score_tier0()

    if has_ops:
        try:
            score_tier1()
        except Exception:
            print(f"  Tier 1 crashed: {traceback.format_exc()}")
            tier_scores["tier1"] = 0.0

        try:
            score_tier2()
        except Exception:
            print(f"  Tier 2 crashed: {traceback.format_exc()}")
            tier_scores["tier2"] = 0.0
    else:
        tier_scores["tier1"] = 0.0
        tier_scores["tier2"] = 0.0

    score_tier3()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(tier_scores.values())
    for tier, score in sorted(tier_scores.items()):
        print(f"  {tier}: {score:.1f}")
    print(f"  -----------")
    print(f"  Total: {total:.1f}/100.0")
    print()
    print(f"SCORE: {total:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
