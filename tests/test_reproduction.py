import unittest, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_scores import main as eval_main, row_z, strict_ranks
import numpy as np

class ScoreOnlyTest(unittest.TestCase):
    def test_strict_ties_count_as_errors(self):
        """Strict silver evaluation: ties with other candidates count as errors."""
        s = np.array([[0.2, 0.2, 0.1], [0.2, 0.3, 0.1]])
        np.testing.assert_array_equal(strict_ranks(s, np.array([0, 1])), [2, 1])

    def test_frozen_reproduction_test_pool(self):
        """Verify baseline, primary k=3, k=10, and validation-selected beta=2 on held-out test."""
        old = sys.argv
        try:
            sys.argv = ['evaluate_scores', '--check', '--out', str(ROOT / 'results/recomputed_test_metrics.json')]
            m = eval_main()
            
            # Baseline
            self.assertEqual(m['baseline']['correct_at_1'], 739)
            self.assertAlmostEqual(m['baseline']['R1'], 739 / 781)
            
            # Primary confirmatory k=3
            k3 = m['csls_train_derived']['3']
            self.assertEqual(k3['correct_at_1'], 749)
            self.assertEqual(k3['fixes'], 14)
            self.assertEqual(k3['regressions'], 4)
            self.assertAlmostEqual(k3['mcnemar_two_sided_exact'], 0.0308837890625)
            
            # Sensitivity k=10
            k10 = m['csls_train_derived']['10']
            self.assertEqual(k10['correct_at_1'], 752)
            self.assertEqual(k10['fixes'], 14)
            self.assertEqual(k10['regressions'], 1)
            
            # Retrospective validation-selected Inverted Softmax beta=2
            beta2 = m['inverted_softmax_train_derived']['2']
            self.assertEqual(beta2['correct_at_1'], 759)
            self.assertEqual(beta2['fixes'], 20)
            self.assertEqual(beta2['regressions'], 0)
            self.assertAlmostEqual(beta2['mcnemar_two_sided_exact'], 1.9073486328125e-06)
        finally:
            sys.argv = old

    def test_validation_selection_end_to_end(self):
        """Verify end-to-end selection on frozen validation matrices: baseline 725, best beta=2 (736)."""
        from val_select import main as val_main
        # Test loading directly
        val = np.load(ROOT / 'data/scores/VAL_scores.npz')
        train_val = np.load(ROOT / 'data/scores/train_VALpool.npz')
        self.assertEqual(val['dice'].shape, (768, 768))
        self.assertEqual(train_val['dice'].shape, (3327, 768))
        self.assertTrue(np.array_equal(val['candidate_ids'], train_val['candidate_ids']))

if __name__ == '__main__':
    unittest.main()
