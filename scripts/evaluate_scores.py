#!/usr/bin/env python3
"""Evaluate BA48 frozen score matrices without access to any newspaper text.

Deterministic top-k, MRR, per-family R@1, McNemar tests and unlocked descriptive
k/beta sensitivity. The original pre-specified primary is k=3 only.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parents[1]


def row_z(x):
    a = np.asarray(x, dtype=np.float64)
    return (a-a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True)+1e-9)


def strict_ranks(x, gold):
    """Rank 1 iff designated score strictly exceeds ALL other candidates."""
    s = np.asarray(x)
    return (s >= s[np.arange(len(gold)), gold, None]).sum(axis=1)


def statistics(ranks):
    return {
        'correct_at_1': int((ranks == 1).sum()),
        'correct_at_5': int((ranks <= 5).sum()),
        'correct_at_10': int((ranks <= 10).sum()),
        'R1': float((ranks == 1).mean()),
        'R5': float((ranks <= 5).mean()),
        'R10': float((ranks <= 10).mean()),
        'MRR_full_pool': float((1.0/ranks).mean()),
    }


def paired_stats(before, after):
    fixed = int(((before > 1) & (after == 1)).sum())
    broken = int(((before == 1) & (after > 1)).sum())
    p = float(binomtest(fixed, fixed+broken, 0.5).pvalue) if fixed+broken else 1.0
    return {'fixes': fixed, 'regressions': broken, 'mcnemar_two_sided_exact': p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT/'data')
    parser.add_argument('--out', type=Path, default=None, help='Optional output JSON path')
    parser.add_argument('--check', action='store_true', help='Assert reported R@1 counts and paired changes')
    args = parser.parse_args()
    data = args.data_dir
    query = np.load(data/'scores/TEST_scores.npz', allow_pickle=False)
    train = np.load(data/'scores/train_TESTpool.npz', allow_pickle=False)
    assert np.array_equal(query['candidate_ids'], train['candidate_ids'])
    qids = list(map(str, query['query_ids']))
    cids = list(map(str, query['candidate_ids']))
    assert len(qids) == 781 and len(set(qids)) == len(qids)
    assert len(cids) == 780 and len(set(cids)) == len(cids)
    with (data/'metadata/test_reference_ids.csv').open(encoding='utf8', newline='') as f:
        refs = list(csv.DictReader(f))
    ref = {r['query_id']: r for r in refs}
    assert len(ref) == 781
    cidx = {c:i for i,c in enumerate(cids)}
    gold = np.array([cidx[ref[q]['gold_candidate_id']] for q in qids], dtype=np.int64)
    families = np.array([ref[q]['doc_family'] for q in qids])
    F = (row_z(query['dice'])+row_z(query['char'])+row_z(query['neural_E1']))/3
    T = (row_z(train['dice'])+row_z(train['char'])+row_z(train['neural_E1']))/3
    baseline_rank = strict_ranks(F, gold)
    out = {
        'status': 'k3 confirmatory; other parameter scans on opened TEST descriptive only',
        'query_count': len(qids), 'candidate_count': len(cids),
        'baseline': statistics(baseline_rank), 'csls_train_derived': {},
        'inverted_softmax_train_derived': {},
    }
    k3_rank = None
    for k in range(1,11):
        d=np.partition(T, -k, axis=0)[-k:].mean(axis=0)
        ranks=strict_ranks(2*F-d, gold)
        role = 'pre_specified_primary' if k == 3 else ('declared_sensitivity' if k == 5 else 'posthoc_descriptive')
        out['csls_train_derived'][str(k)]={'role':role, **statistics(ranks),**paired_stats(baseline_rank,ranks)}
        if k==3:k3_rank=ranks
    for beta in (0.5,1,2,5,10):
        ranks=strict_ranks(beta*F-logsumexp(beta*T, axis=0), gold)
        out['inverted_softmax_train_derived'][str(beta)]={'role':'retrospective_validation_selected_beta2' if beta==2 else 'posthoc_descriptive',**statistics(ranks),**paired_stats(baseline_rank,ranks)}
    assert k3_rank is not None
    by_family=[]
    for fam in sorted(set(families)):
        m=(families==fam)
        by_family.append({'doc_family':fam,'queries':int(m.sum()),'baseline_correct':int((baseline_rank[m]==1).sum()),'k3_correct':int((k3_rank[m]==1).sum())})
    out['k3_by_family']=by_family
    if args.check:
        assert out['baseline']['correct_at_1']==739
        assert out['csls_train_derived']['3']['correct_at_1']==749
        assert out['csls_train_derived']['3']['fixes']==14
        assert out['csls_train_derived']['3']['regressions']==4
        assert out['csls_train_derived']['10']['correct_at_1']==752
        assert out['inverted_softmax_train_derived']['2']['correct_at_1']==759
        assert out['inverted_softmax_train_derived']['2']['fixes']==20
        assert out['inverted_softmax_train_derived']['2']['regressions']==0
        assert len(by_family)==20
        assert sum(x['baseline_correct'] for x in by_family)==739
        assert sum(x['k3_correct'] for x in by_family)==749
        print('PASS: 739 baseline / 749 primary k3 / 752 k10 / 759 Inverted Softmax beta2; 20 families')
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    else:
        print(json.dumps({key:value for key,value in out.items() if key!='k3_by_family'},indent=2,ensure_ascii=False))
    return out

if __name__=='__main__': main()
