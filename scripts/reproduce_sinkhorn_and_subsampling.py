#!/usr/bin/env python3
"""Reproduce transductive Sinkhorn balancing and query-batch subsampling stress test on 780x780.

Runs:
- Transductive Sinkhorn across beta in {0.5, 1, 2, 5, 10} (order: row then column, 10 cycles)
- Query-batch subsampling stress test across fractions {1.0, 0.75, 0.50, 0.25, 0.10}
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EXCLUDED_QUERY_ID = 'Q-e96d59ce6d0cbaf7'


def row_z(x):
    a = np.asarray(x, dtype=np.float32)
    return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + np.float32(1e-9))


def strict(S, g):
    return (S >= S[np.arange(len(g)), g, None]).sum(axis=1) == 1


def sinkhorn(F, beta, order='rowcol', cycles=10):
    """Ordinary-domain Sinkhorn, globally scaled exponentials, uniform marginals."""
    P = np.exp(beta * np.asarray(F, dtype=np.float64) - np.max(beta * F))
    for _ in range(cycles):
        if order == 'rowcol':
            P /= P.sum(axis=1, keepdims=True)
            P /= P.sum(axis=0, keepdims=True)
        elif order == 'colrow':
            P /= P.sum(axis=0, keepdims=True)
            P /= P.sum(axis=1, keepdims=True)
        else:
            raise ValueError(order)
    return P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'data')
    parser.add_argument('--draws', type=int, default=200, help='Number of random subsample draws per fraction')
    parser.add_argument('--seed', type=int, default=20261029, help='Random seed for reproducible subsampling')
    parser.add_argument('--check', action='store_true', help='Assert reported Sinkhorn counts')
    parser.add_argument('--run-subsampling', action='store_true', help='Execute 200-draw subsampling simulation')
    args = parser.parse_args()

    data = args.data_dir
    query = np.load(data / 'scores/TEST_scores.npz', allow_pickle=False)
    train = np.load(data / 'scores/train_TESTpool.npz', allow_pickle=False)

    qids_all = list(map(str, query['query_ids']))
    cids = list(map(str, query['candidate_ids']))

    with (data / 'metadata/test_reference_ids.csv').open(encoding='utf8', newline='') as f:
        refs = list(csv.DictReader(f))
    ref = {r['query_id']: r for r in refs}

    # Retain 780x780 1-to-1 pool
    keep_mask = np.array([q != EXCLUDED_QUERY_ID for q in qids_all])
    qids = [q for q in qids_all if q != EXCLUDED_QUERY_ID]
    cidx = {c: i for i, c in enumerate(cids)}
    gold = np.array([cidx[ref[q]['gold_candidate_id']] for q in qids], dtype=np.int64)

    F = (row_z(query['dice']) + row_z(query['char']) + row_z(query['neural_E1'])) / np.float32(3)
    F = F[keep_mask]
    assert F.shape == (780, 780)

    # 1. Temperature Scan for Sinkhorn
    sinkhorn_results = {}
    for beta in (0.5, 1, 2, 5, 10):
        P_rowcol = sinkhorn(F, beta, order='rowcol')
        corr_rowcol = int(strict(P_rowcol, gold).sum())
        P_colrow = sinkhorn(F, beta, order='colrow')
        corr_colrow = int(strict(P_colrow, gold).sum())
        sinkhorn_results[str(beta)] = {
            'rowcol': corr_rowcol,
            'colrow': corr_colrow,
            'rowcol_R1_pct': round(100 * corr_rowcol / 780, 2),
        }

    print("=== Transductive Sinkhorn Results (780x780) ===")
    for b, res in sinkhorn_results.items():
        print(f"Beta {b:>3s}: row->col = {res['rowcol']}/780 ({res['rowcol_R1_pct']}%), col->row = {res['colrow']}/780")

    if args.check:
        assert sinkhorn_results['5']['rowcol'] == 769, f"Expected 769 for beta=5, got {sinkhorn_results['5']['rowcol']}"
        assert sinkhorn_results['2']['rowcol'] == 764, f"Expected 764 for beta=2, got {sinkhorn_results['2']['rowcol']}"
        print("PASS: Sinkhorn beta=5 achieves 769/780 (98.59%), beta=2 achieves 764/780 (97.95%)")

    # 2. Query Batch Subsampling Simulation (optional full rerun or fast summary)
    if args.run_subsampling:
        print(f"\nRunning {args.draws} random subsample draws per fraction (seed={args.seed})...")
        rng = np.random.default_rng(args.seed)
        fractions = [1.0, 0.75, 0.50, 0.25, 0.10]
        results_table = []
        for frac in fractions:
            nsub = int(round(len(gold) * frac))
            draws = 1 if frac == 1.0 else args.draws
            sink5_acc, sink2_acc = [], []
            for _ in range(draws):
                ix = np.arange(len(gold)) if frac == 1.0 else np.sort(rng.choice(len(gold), size=nsub, replace=False))
                gg = gold[ix]
                sink5_acc.append(100 * float(strict(sinkhorn(F[ix], 5, 'rowcol'), gg).mean()))
                sink2_acc.append(100 * float(strict(sinkhorn(F[ix], 2, 'rowcol'), gg).mean()))
            results_table.append({
                'Fraction': f"{int(frac*100)}%",
                'Queries': nsub,
                'Sinkhorn_beta5_mean': round(float(np.mean(sink5_acc)), 2),
                'Sinkhorn_beta2_mean': round(float(np.mean(sink2_acc)), 2),
            })
        print(pd.DataFrame(results_table).to_string(index=False))


if __name__ == '__main__':
    main()

