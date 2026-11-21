"""Small static/isolated guard tests; no model training."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SRC=Path(__file__).resolve().parents[1]/'scripts'/'run_r12_agnam_isolated_cost.py'

class R12CostGuards(unittest.TestCase):
    def test_script_parses(self):
        ast.parse(SRC.read_text(encoding='utf8'))
    def test_guard_is_before_execution(self):
        txt=SRC.read_text(encoding='utf8')
        assert txt.index('require_no_overwrite(dest)') < txt.index('if not args.execute:')
        assert txt.index('if not args.execute:') < txt.index('dest.mkdir(parents=True)')
    def test_only_full_path_training_function(self):
        txt=SRC.read_text(encoding='utf8')
        self.assertIn('run_reproducible_interaction_discovery(',txt)
        self.assertIn('evaluate_isr_for_discovery(',txt)
        self.assertIn('_train_interaction_model(',txt)
        self.assertNotIn('run_single_synthetic_benchmark(',txt)
    def test_no_benchmark_output_write(self):
        txt=SRC.read_text(encoding='utf8')
        self.assertNotIn('to_csv(',txt)
        self.assertNotIn('results/synthetic/',txt)

if __name__=='__main__':unittest.main()

class R12CostCollectorGuards(unittest.TestCase):
    def test_collector_parses(self):
        p=SRC.parent/'collect_r12_agnam_fullpath_cost.py'
        ast.parse(p.read_text(encoding='utf8'))
    def test_collector_is_read_only(self):
        p=SRC.parent/'collect_r12_agnam_fullpath_cost.py'
        t=p.read_text(encoding='utf8')
        self.assertNotIn('--execute',t)
        self.assertNotIn('run_single_synthetic_benchmark',t)
        self.assertIn('file_sha256',t)
