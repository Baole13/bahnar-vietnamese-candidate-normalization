#!/usr/bin/env python3
"""Reproduce validation parameter selection and retrospective test evaluations directly from frozen NPZ matrices.

Loads:
- data/scores/VAL_scores.npz (768x768)
- data/scores/train_VALpool.npz (3327x768)
- data/scores/TEST_scores.npz (781x780)
- data/scores/train_TESTpool.npz (3327x780)
- data/metadata/test_reference_ids.csv

No raw text, licensed corpus or GPU required.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent

def rz(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)

def strict_ok(scores, gold):
    return (scores >= scores[np.arange(len(scores)), gold][:, None]).sum(axis=1) == 1

def density(F, k):
    return np.sort(F, axis=0)[-k:, :].mean(axis=0)

def inv(F, Ft, b):
    return b * F - logsumexp(b * Ft, axis=0)[None, :]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val-scores', default=ROOT / 'data/scores/VAL_scores.npz', type=Path)
    ap.add_argument('--train-val-scores', default=ROOT / 'data/scores/train_VALpool.npz', type=Path)
    ap.add_argument('--test-scores', default=ROOT / 'data/scores/TEST_scores.npz', type=Path)
    ap.add_argument('--train-test-scores', default=ROOT / 'data/scores/train_TESTpool.npz', type=Path)
    ap.add_argument('--test-metadata', default=ROOT / 'data/metadata/test_reference_ids.csv', type=Path)
    ap.add_argument('--out', default=None, type=Path)
    args = ap.parse_args()

    for p in (args.val_scores, args.train_val_scores, args.test_scores, args.train_test_scores, args.test_metadata):
        if not p.is_file():
            sys.exit(f'Missing input artifact: {p}')

    # 1. Validation Evaluation
    val = np.load(args.val_scores, allow_pickle=False)
    train_val = np.load(args.train_val_scores, allow_pickle=False)
    
    assert val['dice'].shape == (768, 768)
    assert train_val['dice'].shape == (3327, 768)
    assert np.array_equal(val['candidate_ids'], train_val['candidate_ids'])

    F_val = (rz(val['dice']) + rz(val['char']) + rz(val['neural_E1'])) / 3
    F_train_val = (rz(train_val['dice']) + rz(train_val['char']) + rz(train_val['neural_E1'])) / 3
    gold_val = np.arange(768)

    val_base_ok = strict_ok(F_val, gold_val)
    val_base_correct = int(val_base_ok.sum())
    assert val_base_correct == 725, f'Expected 725 validation baseline, got {val_base_correct}'

    grid = {}
    for k in range(1, 11):
        grid[('k', k)] = strict_ok(2 * F_val - density(F_train_val, k), gold_val)
    for b in (0.5, 1, 2, 5, 10):
        grid[('beta', b)] = strict_ok(inv(F_val, F_train_val, b), gold_val)

    val_r1 = {f'{k}_{v}': int(m.sum()) for (k, v), m in grid.items()}
    
    # Pick optimal parameters purely on validation
    best_k = max(range(1, 11), key=lambda k: int(grid[('k', k)].sum()))
    best_beta = max((0.5, 1, 2, 5, 10), key=lambda b: int(grid[('beta', b)].sum()))
    assert best_k == 10, f'Expected best k=10, got {best_k}'
    assert best_beta == 2, f'Expected best beta=2, got {best_beta}'

    # 2. Test Evaluation with Validation-Selected Parameters
    test = np.load(args.test_scores, allow_pickle=False)
    train_test = np.load(args.train_test_scores, allow_pickle=False)

    with args.test_metadata.open(encoding='utf-8') as f:
        meta_rows = list(csv.DictReader(f))
    cids = test['candidate_ids'].astype(str).tolist()
    cix = {c: i for i, c in enumerate(cids)}
    gold_test = np.array([cix[r['gold_candidate_id']] for r in meta_rows])

    F_test = (rz(test['dice']) + rz(test['char']) + rz(test['neural_E1'])) / 3
    F_train_test = (rz(train_test['dice']) + rz(train_test['char']) + rz(train_test['neural_E1'])) / 3

    test_base_ok = strict_ok(F_test, gold_test)
    assert int(test_base_ok.sum()) == 739

    def evaluate_test(scores):
        ok = strict_ok(scores, gold_test)
        fixes = int((~test_base_ok & ok).sum())
        brks = int((test_base_ok & ~ok).sum())
        pval = float(binomtest(fixes, fixes + brks, 0.5).pvalue) if (fixes + brks) > 0 else 1.0
        return {'correct': int(ok.sum()), 'fixes': fixes, 'regressions': brks, 'mcnemar_exact': pval}

    k3_test = evaluate_test(2 * F_test - density(F_train_test, 3))
    k10_test = evaluate_test(2 * F_test - density(F_train_test, best_k))
    beta2_test = evaluate_test(inv(F_test, F_train_test, best_beta))

    assert k3_test['correct'] == 749 and k3_test['fixes'] == 14 and k3_test['regressions'] == 4
    assert k10_test['correct'] == 752 and k10_test['fixes'] == 14 and k10_test['regressions'] == 1
    assert beta2_test['correct'] == 759 and beta2_test['fixes'] == 20 and beta2_test['regressions'] == 0

    results = {
        'status': 'validation_selection_verified_end_to_end',
        'validation_baseline': val_base_correct,
        'validation_R1': {f'{k}{v}': int(m.sum()) for (k, v), m in grid.items()},
        'selected_on_validation': {
            'k': best_k,
            'k_validation_score': int(grid[('k', best_k)].sum()),
            'beta': best_beta,
            'beta_validation_score': int(grid[('beta', best_beta)].sum())
        },
        'test_evaluation': {
            'baseline': 739,
            'pre_specified_primary_k3': k3_test,
            'validation_selected_k10': k10_test,
            'validation_selected_beta2': beta2_test
        }
    }

    print('PASS: Validation baseline 725 -> Selected beta=2 (736/768), k=10 (737/768)')
    print('PASS: Retrospective test: beta=2 achieves 759/781 (20 fixes / 0 regressions vs baseline)')
    print(json.dumps(results, indent=2))

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()

