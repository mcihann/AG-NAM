"""Offline smoke tests of R12 cost batch safety, no training."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_r12_agnam_cost_batch.py'
spec=importlib.util.spec_from_file_location('r12_cost_batch',SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class CostBatchSafetyTests(unittest.TestCase):
    def test_missing_results_are_pending(self):
        with tempfile.TemporaryDirectory() as d:
            p=mod.paths(Path(d),'S2',5)
            self.assertEqual(mod.get_status(p,scenario='S2',realization=5),'pending')
            self.assertIn('r12_s2_05_agnam_fullpath',str(p['measurement']))

    def test_partial_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=mod.paths(Path(d),'S1',2)
            Path(p['summary']).parent.mkdir(parents=True)
            Path(p['summary']).write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Partial run'):
                mod.get_status(p,scenario='S1',realization=2)

    def test_archive_without_underlying_files_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=mod.paths(Path(d),'S3',7)
            Path(p['archive']).write_bytes(b'placeholder')
            with self.assertRaisesRegex(RuntimeError,'Archived result exists'):
                mod.get_status(p,scenario='S3',realization=7)

    def test_no_source_copy_or_existing_file_overwrite(self):
        self.assertIn('--execute',SCRIPT.read_text())
        self.assertIn('subprocess.run(cmd, cwd=root, check=True)',SCRIPT.read_text())
        self.assertNotIn('shutil.rmtree',SCRIPT.read_text())

if __name__=='__main__':unittest.main()
