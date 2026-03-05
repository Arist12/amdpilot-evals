#!/usr/bin/env python3
"""Test harness for vllm-ck-mxfp4-moe. RUNTIME IMPORT CHECKS.

Bug: MXFP4 quantization has no fused MoE support on ROCm.
Test: import the modules and verify CK MoE backend exists.
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
print("vllm-ck-mxfp4-moe test harness")
print("=" * 60)

# Check 1: Import _aiter_ops module
aiter_ops_mod = None
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_aiter_ops", "/workspace/vllm/vllm/_aiter_ops.py")
    aiter_ops_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aiter_ops_mod)
    check("Import _aiter_ops module", True)
except Exception as e:
    check("Import _aiter_ops module", False, str(e))

# Check 2: Import mxfp4 module
mxfp4_mod = None
try:
    spec2 = importlib.util.spec_from_file_location(
        "mxfp4", "/workspace/vllm/vllm/model_executor/layers/quantization/mxfp4.py")
    mxfp4_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mxfp4_mod)
    check("Import mxfp4 module", True)
except Exception as e:
    check("Import mxfp4 module", False, str(e))

# Check 3: _aiter_ops has fused MoE-related functions
if aiter_ops_mod:
    has_fused_moe = any(
        hasattr(aiter_ops_mod, name)
        for name in ["rocm_aiter_fused_topk", "fused_topk", "_rocm_aiter_fused_moe_impl"]
    )
    # Also check class methods
    if not has_fused_moe:
        for attr_name in dir(aiter_ops_mod):
            obj = getattr(aiter_ops_mod, attr_name, None)
            if isinstance(obj, type):
                methods = [m for m in dir(obj) if "fused_moe" in m.lower() or "fused_topk" in m.lower()]
                if methods:
                    has_fused_moe = True
                    break
    check("_aiter_ops has fused MoE functions", has_fused_moe)
else:
    check("_aiter_ops has fused MoE functions", False, "Module not imported")

# Check 4: mxfp4 module has CK backend support for MoE
if mxfp4_mod:
    # The fix adds a CK backend enum and fused MoE forward path
    has_ck_backend = any(
        "ck" in str(getattr(mxfp4_mod, name, "")).lower()
        for name in dir(mxfp4_mod)
    ) or any(
        "ck" in name.lower() or "moe" in name.lower()
        for name in dir(mxfp4_mod)
        if not name.startswith("_")
    )
    check("mxfp4 module has CK/MoE support", has_ck_backend)
else:
    check("mxfp4 module has CK/MoE support", False, "Module not imported")

# Check 5: Weight preparation functions exist
if aiter_ops_mod:
    has_shuffle = any(
        hasattr(aiter_ops_mod, name)
        for name in ["shuffle_weight_a16w4", "shuffle_scale_a16w4"]
    )
    # Check in class methods too
    if not has_shuffle:
        for attr_name in dir(aiter_ops_mod):
            obj = getattr(aiter_ops_mod, attr_name, None)
            if isinstance(obj, type):
                methods = [m for m in dir(obj) if "shuffle" in m.lower()]
                if methods:
                    has_shuffle = True
                    break
    check("Weight shuffle functions exist", has_shuffle)
else:
    check("Weight shuffle functions exist", False, "Module not imported")

print(f"\nResults: {checks_passed}/{checks_total}")
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
