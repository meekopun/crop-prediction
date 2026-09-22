from pathlib import Path
import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.dataset = self.root / "archive" / "yield_df.csv"

    def test_summary_command_outputs_expected_fields(self) -> None:
        env = {"PYTHONPATH": str(self.root / "src")}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "yield_prediction.cli",
                "--data",
                str(self.dataset),
                "summary",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("rows: 28242", result.stdout)
        self.assertIn("areas: 101", result.stdout)

    def test_benchmark_command_reports_missing_optional_dependencies(self) -> None:
        env = {"PYTHONPATH": str(self.root / "src")}
        result = subprocess.run(
            [
                sys.executable,
                "-S",  # Exclude site-packages even when ML dependencies are installed.
                "-m",
                "yield_prediction.cli",
                "--data",
                str(self.dataset),
                "benchmark",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("optional ML dependencies", result.stderr)


if __name__ == "__main__":
    unittest.main()
