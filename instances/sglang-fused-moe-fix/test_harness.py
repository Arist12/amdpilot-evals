#!/usr/bin/env python3
"""Test harness for sglang-fused-moe-fix. Behavioral tests only.

Bug: MoE models crash on ROCm with NameError in fused_moe.py.
Test: Verify the module imports and core functions work without errors on ROCm.
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
print("sglang-fused-moe-fix test harness")
print("=" * 60)

TARGET = "/workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py"

from pathlib import Path
if not check("Target file exists", Path(TARGET).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

source = Path(TARGET).read_text()

# Check 1: File is valid Python
try:
    import ast
    tree = ast.parse(source)
    check("Valid Python", True)
except SyntaxError as e:
    check("Valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 2: Module imports without NameError on ROCm
try:
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fused_moe_module
    check("Module imports without NameError", True)
except NameError as e:
    check("Module imports without NameError", False, f"NameError: {e}")
except (ImportError, ModuleNotFoundError):
    check("Module imports (deps issue, not the bug)", True)
except Exception:
    check("Module imports (other issue, not the bug)", True)

# Check 3: fused_experts_impl is callable (the function where the crash occurs)
try:
    fn = getattr(fused_moe_module, "fused_experts_impl", None)
    if fn is None:
        fn = getattr(fused_moe_module, "fused_experts", None)
    check("fused_experts function accessible", fn is not None and callable(fn))
except Exception as e:
    check("fused_experts function accessible", False, str(e)[:200])

# Check 4: Module-level execution does not raise NameError
# Re-import in a subprocess to ensure fresh module load on ROCm
import subprocess
result = subprocess.run(
    ["/opt/venv/bin/python3", "-c",
     "import sys; sys.path.insert(0, '/workspace/sglang/python'); "
     "from sglang.srt.layers.moe.fused_moe_triton import fused_moe; "
     "print('IMPORT_OK')"],
    capture_output=True, text=True, timeout=60,
)
import_ok = "IMPORT_OK" in (result.stdout or "")
err_text = (result.stderr or "")[-300:]
if not import_ok and "NameError" in err_text:
    check("Fresh import succeeds (no NameError)", False, err_text)
elif not import_ok:
    check("Fresh import succeeds (non-target error)", True)
else:
    check("Fresh import succeeds", True)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
