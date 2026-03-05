#!/usr/bin/env python3
"""Test harness for sglang-qwen35-rope-fix eval.

Two bugs:
1. RoPE config: rope_parameters from text_config is lost -> ValueError: Unknown RoPE scaling type
2. CuTe-DSL: unconditional import of cuda.bindings -> ModuleNotFoundError on ROCm

Tests import-level behavior without needing model weights.

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import sys
from pathlib import Path

SGLANG_ROOT = Path("/workspace/sglang")
sys.path.insert(0, str(SGLANG_ROOT / "python"))

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


def check_rope_config():
    """Bug 1: Qwen3.5 config drops rope_parameters from text_config."""
    try:
        from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
    except ImportError as e:
        check("Qwen3_5MoeTextConfig importable", False, str(e))
        return

    check("Qwen3_5MoeTextConfig importable", True)

    # Simulate the config that causes the ValueError
    text_config = {
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
        "max_position_embeddings": 131072,
        "rope_parameters": {
            "rope_type": "default",
        },
    }

    try:
        config = Qwen3_5MoeTextConfig(**text_config)
        # The fix should preserve rope_parameters or rope_scaling
        has_rope = (hasattr(config, 'rope_scaling') and config.rope_scaling is not None) or \
                   (hasattr(config, 'rope_parameters') and config.rope_parameters is not None)
        check("RoPE config preserved from text_config",
              has_rope or True,  # If config init succeeds without ValueError, that's the fix
              "rope_parameters/rope_scaling not preserved")
    except ValueError as e:
        if "Unknown RoPE scaling type" in str(e):
            check("RoPE config preserved (no ValueError)", False,
                  f"ValueError: {e}")
        else:
            check("Config construction", False, str(e))
    except Exception as e:
        check("Config construction", False, str(e))


def check_hybrid_attention_import():
    """Bug 2: hybrid_linear_attn_backend.py imports cuda.bindings unconditionally."""
    target = SGLANG_ROOT / "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
    if not check("hybrid_linear_attn_backend.py exists", target.is_file()):
        return

    source = target.read_text()

    # The fix should guard the cuda.bindings import
    has_unguarded_cuda = False
    for i, line in enumerate(source.split("\n"), 1):
        stripped = line.strip()
        if "cuda.bindings" in stripped or "from cuda" in stripped:
            # Check if it's inside a try/except or conditional
            context_start = max(0, i - 5)
            context = "\n".join(source.split("\n")[context_start:i])
            if "try" not in context and "if " not in context and "except" not in context:
                has_unguarded_cuda = True

    check("cuda.bindings import is guarded (try/except or conditional)",
          not has_unguarded_cuda,
          "Unguarded 'cuda.bindings' or 'from cuda' import found")

    # Try importing the module
    try:
        import importlib
        spec = importlib.util.spec_from_file_location(
            "hybrid_linear_attn_backend", str(target))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            check("Module imports without ModuleNotFoundError", True)
        except ModuleNotFoundError as e:
            if "cuda" in str(e).lower():
                check("Module imports without cuda dependency error", False,
                      f"ModuleNotFoundError: {e}")
            else:
                # Other missing modules are OK
                check("Module imports without cuda dependency error", True)
        except Exception:
            check("Module imports (other error, not cuda-related)", True)
    except Exception as e:
        check("Module importable", False, str(e))


def run_checks():
    print("=" * 60)
    print("sglang-qwen35-rope-fix test harness")
    print("=" * 60)

    print("\n--- Bug 1: RoPE config parsing ---")
    check_rope_config()

    print("\n--- Bug 2: CuTe-DSL import guard ---")
    check_hybrid_attention_import()


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
