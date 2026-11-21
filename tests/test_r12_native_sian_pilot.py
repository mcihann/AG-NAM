"""Zero-training R12 contracts. No GPU, network or SIAN training required."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest
import numpy as np

# Import only the pure functions from the actual delivered script, so tests
# also run on machines without PyTorch or the full AG-NAM runtime installed.
SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "run_r12_sian_pairwise_pilot.py"
parsed = ast.parse(SOURCE.read_text(encoding="utf-8"))
wanted = {"pair_structure", "hash_ids"}
functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
namespace = {"np": np, "hashlib": hashlib}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)


class R12NativeSIANContracts(unittest.TestCase):
    def test_pair_structure_fis_is_pairwise_and_canonical(self):
        terms, pairs = namespace["pair_structure"](
            [(), (0,), (1,), (3, 1), (3, 1)], ("x1", "x2", "x3", "x4")
        )
        self.assertEqual(terms, ((0,), (1,), (1, 3)))
        self.assertEqual(pairs, (("x2", "x4"),))

    def test_reject_higher_order_interactions(self):
        with self.assertRaises(ValueError):
            namespace["pair_structure"]([(1, 2, 3)], ("x1", "x2", "x3", "x4"))

    def test_reject_invalid_indices(self):
        for invalid in [[(1, 1)], [(0, 8)], [(-1, 2)]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                namespace["pair_structure"](invalid, ("x1", "x2", "x3", "x4"))

    def test_split_hash_is_order_sensitive(self):
        f = namespace["hash_ids"]
        self.assertEqual(f(np.array([1, 2, 3])), f([1, 2, 3]))
        self.assertNotEqual(f([1, 2, 3]), f([3, 2, 1]))

    def test_script_requires_execute_and_dedicated_output(self):
        src = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"--execute"', src)
        self.assertIn('if not args.execute:', src)
        self.assertIn('r12_native_sian_pairwise', src)
        self.assertIn('native_select(', src)
        self.assertIn('dnn_on_or_off=False', src)
        self.assertIn('Native SIAN FIS returned no selectable terms', src)


if __name__ == "__main__":
    unittest.main()
