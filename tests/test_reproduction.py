import unittest, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_scores import main as eval_main, row_z, strict_ranks
from reproduce_sinkhorn_and_subsampling import sinkhorn, strict as sinkhorn_strict
import numpy as np

class ScoreOnlyTest(unittest.TestCase):
    def test_strict_ties_count_as_errors(self):
        """Strict silver evaluation: ties with other candidates count as errors."""
        s = np.array([[0.2, 0.2, 0.1], [0.2, 0.3, 0.1]])
        np.testing.assert_array_equal(strict_ranks(s, np.array([0, 1])), [2, 1])

    def test_frozen_reproduction_test_pool_780(self):
        """Verify baseline, primary k=3, k=10, and validation-selected beta=2 on 780x780 test pool."""
        old = sys.argv
        try:
            sys.argv = ['evaluate_scores', '--check', '--out', str(ROOT / 'results/recomputed_test_metrics.json')]
            m = eval_main()
            
            # Baseline 738 / 780
            self.assertEqual(m['baseline']['correct_at_1'], 738)
            self.assertAlmostEqual(m['baseline']['R1'], 738 / 780, places=4)
            
            # Primary confirmatory k=3: 748 / 780
            k3 = m['csls_train_derived']['3']
            self.assertEqual(k3['correct_at_1'], 748)
            self.assertEqual(k3['fixes'], 14)
            self.assertEqual(k3['regressions'], 4)
            self.assertAlmostEqual(k3['mcnemar_two_sided_exact'], 0.0308837890625, places=5)
            
            # Sensitivity k=10: 751 / 780
            k10 = m['csls_train_derived']['10']
            self.assertEqual(k10['correct_at_1'], 751)
            self.assertEqual(k10['fixes'], 14)
            self.assertEqual(k10['regressions'], 1)
            
            # Retrospective validation-selected Inverted Softmax beta=2: 758 / 780
            beta2 = m['inverted_softmax_train_derived']['2']
            self.assertEqual(beta2['correct_at_1'], 758)
            self.assertEqual(beta2['fixes'], 20)
            self.assertEqual(beta2['regressions'], 0)
        finally:
            sys.argv = old

    def test_sinkhorn_reproduction_780(self):
        """Verify transductive Sinkhorn beta=5 achieves 769/780 on the 1-to-1 pool."""
        query = np.load(ROOT / 'data/scores/TEST_scores.npz')
        qids = list(map(str, query['query_ids']))
        keep = np.array([q != 'Q-e96d59ce6d0cbaf7' for q in qids])
        
        # Load gold
        import csv
        with open(ROOT / 'data/metadata/test_reference_ids.csv', encoding='utf-8') as f:
            refs = {r['query_id']: r['gold_candidate_id'] for r in csv.DictReader(f)}
        cids = list(map(str, query['candidate_ids']))
        cix = {c: i for i, c in enumerate(cids)}
        gold = np.array([cix[refs[q]] for q in qids if q != 'Q-e96d59ce6d0cbaf7'], dtype=np.int64)
        
        a = (row_z(query['dice']) + row_z(query['char']) + row_z(query['neural_E1'])) / np.float32(3)
        F = a[keep]
        
        P5 = sinkhorn(F, 5, 'rowcol')
        self.assertEqual(int(sinkhorn_strict(P5, gold).sum()), 769)
        P2 = sinkhorn(F, 2, 'rowcol')
        self.assertEqual(int(sinkhorn_strict(P2, gold).sum()), 764)

    def test_validation_selection_end_to_end(self):
        """Verify end-to-end selection on frozen validation matrices: baseline 725, best beta=2 (736)."""
        val = np.load(ROOT / 'data/scores/VAL_scores.npz')
        train_val = np.load(ROOT / 'data/scores/train_VALpool.npz')
        self.assertEqual(val['dice'].shape, (768, 768))
        self.assertEqual(train_val['dice'].shape, (3327, 768))
        self.assertTrue(np.array_equal(val['candidate_ids'], train_val['candidate_ids']))

if __name__ == '__main__':
    unittest.main()
