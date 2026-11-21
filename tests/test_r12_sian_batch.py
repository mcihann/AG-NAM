"""Fast, zero-training checks of R12 batch orchestration and audit logic."""
from __future__ import annotations
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class R12BatchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = load("run_r12_sian_batch")
        cls.audit = load("collect_r12_sian_batch")

    def test_realization_validation(self):
        self.assertEqual(len(self.batch.scenario_range("S1", 2, 20)), 19)
        for scenario, first, last in [("S0", 1, 2), ("S1", 0, 1),
                                      ("S2", 21, 22), ("S3", 5, 4)]:
            with self.subTest(scenario=scenario, first=first, last=last):
                with self.assertRaises(ValueError):
                    self.batch.scenario_range(scenario, first, last)

    def test_no_automatic_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d, stage, cost = self.batch.run_paths(root, "S1", 1)
            self.assertEqual(self.batch.classify(root, "S1", 1), "new")
            stage.mkdir(parents=True)
            self.assertEqual(self.batch.classify(root, "S1", 1), "inprogress_or_failed")
            stage.rmdir()
            d.mkdir(parents=True)
            self.assertEqual(self.batch.classify(root, "S1", 1), "partial")

    def test_complete_result_requires_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d, stage, cost = self.batch.run_paths(root, "S1", 1)
            d.mkdir(parents=True)
            cost.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"status": "complete", "scenario": "S1",
                                                        "realization": 1}), encoding="utf-8")
            (cost / "measurement.json").write_text(json.dumps({"experiment_run_status": "complete",
                                                                "returncode": 0}), encoding="utf-8")
            self.assertEqual(self.batch.classify(root, "S1", 1), "partial")
            for filename in ("selected_pairs.csv", "test_predictions.csv", "sian_state_dict.pt"):
                (d / filename).write_text("dummy", encoding="utf-8")
            self.assertEqual(self.batch.classify(root, "S1", 1), "complete")

    def test_dry_run_without_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "run_r12_sian_pairwise_pilot.py").write_bytes(b"pilot")
            (scripts / "r12_profile_command.py").write_text("pass")
            with self.assertRaises(RuntimeError):
                self.batch.validate_locked_source(root)
            # No training is invoked by the pure dry-run command builder.
            self.assertIn("--execute", self.batch.command("S1", 2))

    def test_aggregates_and_jaccard(self):
        one = {"scenario": "S1", "test_auroc": 0.8, "test_auprc": 0.7,
               "precision": .5, "recall": 1., "f1": 2 / 3,
               "n_selected_pairs": 6, "tp": 3, "fp": 3, "fn": 0,
               "wall_seconds_external": 100., "peak_rss_mib_sampled": 500.,
               "peak_gpu_allocated_mib": 20., "selected_pairs": [("x1", "x2")]}
        two = {**one, "test_auroc": .9, "selected_pairs": [("x2", "x3")]}
        s = self.audit.summarize([one, two])[0]
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["test_auroc_mean"], .85)
        self.assertEqual(s["mean_pairwise_jaccard_descriptive"], 0.0)


if __name__ == "__main__":
    unittest.main()
