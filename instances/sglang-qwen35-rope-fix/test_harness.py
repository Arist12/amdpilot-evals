#!/usr/bin/env python3
"""Test harness for sglang-qwen35-rope-fix. RUNTIME CHECKS.

Bug 1: Qwen3.5 config loses rope_parameters -> ValueError
Bug 2: hybrid_linear_attn_backend imports cuda.bindings unconditionally -> ModuleNotFoundError
"""
import sys
sys.path.insert(0, "/workspace/sglang/python")

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

print("=" * 60)
print("sglang-qwen35-rope-fix test harness")
print("=" * 60)

# === Bug 1: RoPE config ===
print("\n--- Bug 1: RoPE config ---")
try:
    from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
    check("Import Qwen3_5MoeTextConfig", True)

    # This config simulates what HuggingFace sends with rope_parameters
    config = Qwen3_5MoeTextConfig(
        hidden_size=4096, num_attention_heads=64,
        num_key_value_heads=4, intermediate_size=12288,
        max_position_embeddings=131072,
        rope_parameters={"rope_type": "default"},
    )
    check("Config construction with rope_parameters (no ValueError)", True)

except ValueError as e:
    if "RoPE" in str(e) or "rope" in str(e).lower():
        check("Config construction with rope_parameters (no ValueError)", False,
              f"ValueError: {e}")
    else:
        check("Config construction", False, str(e))
except ImportError as e:
    check("Import Qwen3_5MoeTextConfig", False, str(e))
except Exception as e:
    check("Config construction", False, f"{type(e).__name__}: {e}")

# === Bug 2: cuda.bindings import ===
print("\n--- Bug 2: CuTe-DSL import ---")
TARGET = "/workspace/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"

from pathlib import Path
if not Path(TARGET).is_file():
    check("hybrid_linear_attn_backend.py exists", False)
else:
    check("hybrid_linear_attn_backend.py exists", True)

    # Try importing the module -- on ROCm without cuda.bindings, this should not crash
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hybrid_linear_attn_backend", TARGET)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        check("Module imports without cuda.bindings crash", True)
    except ModuleNotFoundError as e:
        if "cuda" in str(e).lower():
            check("Module imports without cuda.bindings crash", False,
                  f"ModuleNotFoundError: {e}")
        else:
            # Other missing modules are not this bug
            check("Module imports (other dep missing, not cuda)", True)
    except Exception as e:
        check("Module imports", False, f"{type(e).__name__}: {e}")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
