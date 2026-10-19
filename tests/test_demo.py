"""Offline regression checks for the exported scientific example."""
import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo", ROOT / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = demo.load_bundle()

    def test_forecast_change_rejects_mismatched_number(self):
        evidence = copy.deepcopy(self.bundle["evidence"])
        evidence["stations"][0]["forecasts"][0]["difference"] = "999,0"
        with self.assertRaisesRegex(ValueError, "change"):
            demo.validate_evidence(evidence)

    def test_interval_rejects_reversed_bounds(self):
        evidence = copy.deepcopy(self.bundle["evidence"])
        evidence["stations"][0]["forecasts"][0]["lower"] = "999,0"
        with self.assertRaisesRegex(ValueError, "interval"):
            demo.validate_evidence(evidence)

    def test_fact_reference_must_match_evidence(self):
        report = copy.deepcopy(self.bundle["reports"]["evolution"])
        report["evidence_checks"][0]["value"] = "invented"
        schema = demo.read_json(ROOT / "examples/contracts/evolution.json")
        with self.assertRaisesRegex(ValueError, "fact reference"):
            demo.validate_report(report, schema, self.bundle["evidence"]["facts"])

    def test_report_text_is_escaped_in_html(self):
        bundle = copy.deepcopy(self.bundle)
        bundle["reports"]["evolution"]["title"] = '<script>alert("test")</script>'
        page = demo.render_html(bundle)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("AI-assisted review", page)
        self.assertIn("Not requested", page)

    def test_isolated_copy_needs_no_packages_or_original_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / "standalone"
            package.mkdir()
            for folder in ("examples", "results"):
                shutil.copytree(ROOT / folder, package / folder)
            shutil.copy2(ROOT / "demo.py", package / "demo.py")
            # -I ignores user environment/site paths; -S disables installed packages.
            completed = subprocess.run([sys.executable, "-I", "-S", "-B", str(package / "demo.py")],
                                       cwd=base, capture_output=True, encoding="utf-8", timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((package / "demo_output/index.html").is_file())
            self.assertIn("No model training", completed.stdout)
            evidence = package / "examples/evidence.json"
            evidence.write_bytes(evidence.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                demo.verify_files(package)


if __name__ == "__main__":
    unittest.main()
