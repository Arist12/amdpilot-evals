import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.validation_tools import (  # noqa: E402
    build_validation_spec,
    classify_tier,
    generate_deterministic_checks,
    normalize_validation_commands,
)


class ValidationToolsTests(unittest.TestCase):
    def test_normalize_validation_commands_filters_descriptions(self) -> None:
        self.assertEqual(
            normalize_validation_commands("N/A\npytest tests/test_x.py"),
            ["pytest tests/test_x.py"],
        )

    def test_normalize_validation_commands_rewrites_filecheck(self) -> None:
        self.assertEqual(
            normalize_validation_commands("FileCheck on test/Conversion/amd/foo.mlir"),
            ["cd /workspace/build && lit -v ../test/Conversion/amd/foo.mlir"],
        )

    def test_normalize_validation_commands_picks_first_or_alternative(self) -> None:
        self.assertEqual(
            normalize_validation_commands("pytest tests/test_a.py OR pytest tests/test_b.py"),
            ["pytest tests/test_a.py"],
        )

    def test_classify_tier(self) -> None:
        self.assertEqual(classify_tier(["pytest x.py"], []), 1)
        self.assertEqual(classify_tier([], ["x.py"]), 2)
        self.assertEqual(classify_tier([], []), 3)

    def test_generate_deterministic_checks_uses_diff_hint(self) -> None:
        checks = generate_deterministic_checks("example/repo", "ground_truth/foo.diff")
        self.assertIn("agent made changes", checks[0])
        self.assertTrue(checks[-1].startswith("cd /workspace && git diff --stat"))

    def test_build_validation_spec_marks_kimi_sglang_bootstrap_supported(self) -> None:
        pr = {
            "repo": "sgl-project/sglang",
            "pr_number": 19228,
            "title": "[AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning",
            "problem": "Kimi K2.5 fused_moe_triton use default config so the performance is poor.",
            "test_commands": "python3 benchmark/gsm8k/bench_sglang.py --host http://127.0.0.1 --port 30000 --num-questions 2000 --parallel 2000 --num-shots 8",
        }
        spec = build_validation_spec(pr)
        self.assertEqual(spec["tier"], 1)
        self.assertEqual(spec["mode"], "tests")
        self.assertIn("model_server_bootstrap", spec)
        self.assertTrue(spec["model_server_bootstrap"]["supported"])


if __name__ == "__main__":
    unittest.main()
