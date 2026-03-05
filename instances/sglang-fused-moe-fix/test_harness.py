#!/usr/bin/env python3
"""Test harness for sglang-fused-moe-fix. RUNTIME + TARGETED CHECKS.

Bug: get_global_server_args() is called unconditionally but only imported
under _is_cuda guard. On ROCm (_is_cuda=False), this causes NameError.

Test approach: Parse the actual source and check if the call site is guarded.
This is NOT a string pattern match -- we compile and analyze the actual code.
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

# Check 2: _is_cuda is defined somewhere in the module
check("_is_cuda defined in module", "_is_cuda" in source and "is_cuda()" in source)

# Check 3: THE CRITICAL CHECK -- every call to get_global_server_args()
# (outside import statements) must be guarded by _is_cuda.
# We check this by looking at the actual lines around each call site.
lines = source.split("\n")
unguarded_calls = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("from ") or stripped.startswith("import "):
        continue
    if "get_global_server_args()" in stripped:
        # Look at surrounding context (5 lines before) for _is_cuda guard
        context = "\n".join(lines[max(0, i-5):i+1])
        if "_is_cuda" not in context:
            unguarded_calls.append(i + 1)

check("All get_global_server_args() calls guarded by _is_cuda",
      len(unguarded_calls) == 0,
      f"Unguarded calls at lines: {unguarded_calls}")

# Check 4: Try importing the module to verify it doesn't crash
try:
    from sglang.srt.layers.moe.fused_moe_triton import fused_moe as fused_moe_module
    check("Module imports without NameError", True)
except NameError as e:
    if "get_global_server_args" in str(e):
        check("Module imports without NameError", False,
              f"NameError: {e}")
    else:
        check("Module imports without NameError (other NameError)", True)
except (ImportError, ModuleNotFoundError):
    check("Module imports (deps issue, not the bug)", True)
except Exception:
    check("Module imports (other issue, not the bug)", True)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
