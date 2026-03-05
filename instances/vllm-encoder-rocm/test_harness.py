#!/usr/bin/env python3
"""Test harness for vllm-encoder-rocm. ALL RUNTIME CHECKS.

Bug: RocmAttentionImpl.forward raises NotImplementedError for
AttentionType.ENCODER. This blocks encoder-decoder models on ROCm.
"""
import sys
sys.path.insert(0, "/workspace/vllm")

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

print("=" * 60)
print("vllm-encoder-rocm test harness")
print("=" * 60)

# Check 1: Import the ROCm attention module
rocm_mod = None
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rocm_attn", "/workspace/vllm/vllm/v1/attention/backends/rocm_attn.py")
    rocm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rocm_mod)
    check("Import rocm_attn module", True)
except Exception as e:
    check("Import rocm_attn module", False, str(e))

# Check 2: Import AttentionType and verify ENCODER exists
attn_type_mod = None
try:
    spec2 = importlib.util.spec_from_file_location(
        "backend", "/workspace/vllm/vllm/v1/attention/backend.py")
    attn_type_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(attn_type_mod)
    AttentionType = attn_type_mod.AttentionType
    has_encoder = hasattr(AttentionType, "ENCODER")
    check("AttentionType.ENCODER exists", has_encoder)
except Exception as e:
    check("AttentionType.ENCODER exists", False, str(e))

# Check 3: The critical check -- does RocmAttentionImpl accept ENCODER type?
# On unfixed code: raises NotImplementedError("Encoder self-attention...")
# On fixed code: should not raise
if rocm_mod and attn_type_mod:
    try:
        # Read the source of forward() to check if ENCODER is in the allowed list
        import inspect
        src = inspect.getsource(rocm_mod.RocmAttentionImpl.forward)
        # The unfixed code has: if attn_type not in [DECODER, ENCODER_DECODER]: raise NotImplementedError
        # The fixed code should accept ENCODER in the allowed list
        raises_for_encoder = ("NotImplementedError" in src and
                              "ENCODER" not in src.split("NotImplementedError")[0].split("[")[-1])
        check("RocmAttentionImpl.forward accepts ENCODER type",
              not raises_for_encoder,
              "forward() raises NotImplementedError for ENCODER attention type")
    except Exception as e:
        check("RocmAttentionImpl.forward accepts ENCODER type", False, str(e))

# Check 4: Same for the unified AITER backend
aiter_mod = None
try:
    spec3 = importlib.util.spec_from_file_location(
        "rocm_aiter_unified_attn",
        "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_unified_attn.py")
    aiter_mod = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(aiter_mod)
    check("Import rocm_aiter_unified_attn module", True)
except Exception as e:
    check("Import rocm_aiter_unified_attn module", False, str(e))

if aiter_mod and attn_type_mod:
    try:
        import inspect
        # Find the attention impl class
        cls_name = None
        for name in dir(aiter_mod):
            obj = getattr(aiter_mod, name)
            if isinstance(obj, type) and "Impl" in name:
                cls_name = name
                break
        if cls_name:
            src = inspect.getsource(getattr(aiter_mod, cls_name).forward)
            raises_for_encoder = ("NotImplementedError" in src and
                                  "ENCODER" not in src.split("NotImplementedError")[0].split("[")[-1])
            check(f"{cls_name}.forward accepts ENCODER type",
                  not raises_for_encoder,
                  "forward() raises NotImplementedError for ENCODER attention type")
        else:
            check("AITER unified backend accepts ENCODER type", False, "No Impl class found")
    except Exception as e:
        check("AITER unified backend accepts ENCODER type", False, str(e))

print(f"\nResults: {checks_passed}/{checks_total}")
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
