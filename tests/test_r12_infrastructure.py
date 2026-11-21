"""Smoke tests: no model training or benchmark data access."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R12InfrastructureTests(unittest.TestCase):
    def test_profiler_dry_run_does_not_create_output(self):
        mod = load_script('r12_profile_command')
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'never_create'
            result = mod.run(['--label', 'dry', '--output-root', str(dest),
                              '--', sys.executable, '-c', 'raise RuntimeError("should not run")'])
            self.assertEqual(result, 0)
            self.assertFalse(dest.exists())

    def test_profiler_executes_and_writes_manifest(self):
        mod = load_script('r12_profile_command')
        with tempfile.TemporaryDirectory() as tmp:
            result = mod.run(['--label', 'tiny', '--execute', '--output-root', tmp, '--',
                              sys.executable, '-c', 'print("r12_check_ok")'])
            self.assertEqual(result, 0)
            data = json.loads((Path(tmp) / 'tiny' / 'measurement.json').read_text())
            self.assertEqual(data['returncode'], 0)
            self.assertEqual(data['experiment_run_status'], 'complete')
            self.assertGreaterEqual(data['elapsed_wall_seconds'], 0.0)
            self.assertIn('r12_check_ok', (Path(tmp) / 'tiny' / 'command_output.log').read_text())

    def test_avoid_duplicate_label_overwrite(self):
        mod = load_script('r12_profile_command')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'tiny').mkdir()
            with self.assertRaises(SystemExit):
                mod.run(['--label', 'tiny', '--execute', '--output-root', tmp,
                         '--', sys.executable, '-c', 'print("bad")'])

    def test_source_exclusions(self):
        mod = load_script('collect_r12_preflight')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'src' / 'agnam').mkdir(parents=True)
            (root / 'src' / 'agnam' / 'example.py').write_text('def build_synthetic_data(x): return x')
            (root / 'src' / 'agnam' / 'secrets.py').write_text('do_not_copy = "hi"')
            (root / 'results').mkdir()
            (root / 'results' / 'hidden.py').write_text('no')
            files = [p.name for p in mod.source_files(root)]
            self.assertIn('example.py', files)
            self.assertNotIn('secrets.py', files)
            self.assertNotIn('hidden.py', files)


if __name__ == '__main__':
    unittest.main()
