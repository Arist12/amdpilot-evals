import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.registry_tools import (
    classify_gpu_required,
    key_files_to_list,
    manifest_path,
    normalize_test_commands,
    parse_pr_ref,
    pr_slug,
)


class RegistryToolsTests(unittest.TestCase):
    def test_parse_pr_ref_accepts_url(self) -> None:
        repo, pr_number = parse_pr_ref(
            "https://github.com/sgl-project/sglang/pull/18903"
        )
        self.assertEqual(repo, "sgl-project/sglang")
        self.assertEqual(pr_number, 18903)

    def test_parse_pr_ref_accepts_short_form(self) -> None:
        repo, pr_number = parse_pr_ref("sgl-project/sglang/18903")
        self.assertEqual(repo, "sgl-project/sglang")
        self.assertEqual(pr_number, 18903)

    def test_normalize_test_commands_drops_na(self) -> None:
        self.assertEqual(normalize_test_commands("N/A"), [])

    def test_normalize_test_commands_splits_multiline_text(self) -> None:
        self.assertEqual(
            normalize_test_commands("pytest tests/test_a.py\npython test_b.py"),
            ["pytest tests/test_a.py", "python test_b.py"],
        )

    def test_key_files_to_list_handles_csv(self) -> None:
        self.assertEqual(
            key_files_to_list("a.py, b.py, c.py"),
            ["a.py", "b.py", "c.py"],
        )

    def test_gpu_required_uses_repo_and_command_hints(self) -> None:
        self.assertTrue(classify_gpu_required("ROCm/aiter", []))
        self.assertTrue(
            classify_gpu_required(
                "example/repo",
                ["python bench.py --device gpu --arch gfx942"],
            )
        )
        self.assertFalse(classify_gpu_required("example/repo", ["pytest tests/test_x.py"]))

    def test_pr_slug_preserves_case_but_sanitizes(self) -> None:
        self.assertEqual(pr_slug("sgl-project/sglang"), "sgl-project_sglang")

    def test_manifest_path_prefers_repo_relative_paths(self) -> None:
        project_root = Path("/repo")
        self.assertEqual(
            manifest_path(Path("/repo/diffs/x.diff"), project_root),
            "diffs/x.diff",
        )
        self.assertEqual(
            manifest_path(Path("/tmp/diffs/x.diff"), project_root),
            "/tmp/diffs/x.diff",
        )


if __name__ == "__main__":
    unittest.main()
