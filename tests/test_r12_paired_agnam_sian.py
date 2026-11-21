"""Fast R12 audit helpers tests; no training."""
import importlib.util
from pathlib import Path
import tempfile
import json
import csv
import unittest

SRC = Path(__file__).resolve().parents[1] / 'scripts' / 'collect_r12_paired_agnam_sian.py'
spec = importlib.util.spec_from_file_location('r12_paired', SRC)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class TestPairedAudit(unittest.TestCase):
    def test_truth_strict(self):
        self.assertTrue(audit.truth('True'))
        self.assertFalse(audit.truth('false'))
        with self.assertRaises(ValueError): audit.truth('maybe')
    def test_f1(self):
        self.assertAlmostEqual(audit.recovered_f1(3,12,0), 1/3)
        self.assertEqual(audit.recovered_f1(0,0,3),0.0)
    def test_summary(self):
        rows=[{'scenario':'S1', **{m: float(i) for m in audit.METRICS}} for i in (1,2)]
        s=audit.summarize(rows)
        self.assertEqual(s['n'],2)
        self.assertAlmostEqual(s['sian_auroc_mean'],1.5)
        self.assertEqual(s['sian_wins_auroc'],2)
    def test_missing_data_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                audit.check_one(Path(t), 'S1', 1)

    def test_real_data_sian_with_constructed_agnam_record(self):
        """Exercises the complete check_one contract using genuine SIAN R01 bytes
        alongside a synthetic, internally consistent AG-NAM checkpoint."""
        fixture = Path(__file__).resolve().parent/'fixtures'/'r12_s1_01'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original='results/revision/r12_native_sian_pairwise/S1/realization_01/'
            for name in ('summary.json','selected_pairs.csv','test_predictions.csv'):
                target=root/original/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((fixture/original/name).read_bytes())
            cost='results/revision/r12_cost_audit/r12_s1_01_sian/measurement.json'
            dest=root/cost;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((fixture/cost).read_bytes())
            ag=root/'results/synthetic/S1/realization_01.csv';ag.parent.mkdir(parents=True,exist_ok=True)
            summary=json.loads((fixture/original/'summary.json').read_text(encoding='utf-8'))
            row={'scenario':'S1','realization_index':0,'dataset_seed':summary['data_seed'],
                 'outer_split_seed':summary['split_seeds']['outer'],
                 'final_split_seed':summary['split_seeds']['final'],
                 'discovery_base_seed':summary['training_seeds']['reference'],
                 'final_model_seed':summary['training_seeds']['final'],
                 'n_true_interactions':3,'n_isr_retained':1,
                 'isr_precision':1.,'isr_recall':1/3,'isr_f1':.5,
                 'agnam_auroc':.86,'agnam_auprc':.85}
            with ag.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerow(row)
            pair=root/'results/synthetic/S1/realization_01_pairs.csv'
            with pair.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['feature_j','feature_k','is_true','isr_retained'])
                w.writeheader()
                for pa,pb in [('x3','x4'),('x5','x6'),('x8','x9')]:
                    w.writerow({'feature_j':pa,'feature_k':pb,'is_true':True,
                                'isr_retained':pa=='x3'})
            rec,src=audit.check_one(root,'S1',1)
            self.assertEqual(len(src),6)
            self.assertEqual(rec['agnam_n_isr_retained'],1)
            self.assertAlmostEqual(rec['sian_selected_f1'],1/3)
            self.assertAlmostEqual(rec['delta_sian_minus_agnam_auroc'],
                                   summary['test_predictive_metrics']['test_auroc']-.86)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t) / 'existing.zip';p.write_bytes(b'x')
            with self.assertRaises(FileExistsError): audit.collect(Path(t),'S1',p)

if __name__=='__main__':
    unittest.main()
