# E11 evaluation protocol

**Information regime:** fixed Vietnamese candidate pool. Each Bahnar query is ranked against the same 780 candidates. The target sentence for two queries shares an ID. Candidate density is estimated from the 3,327 *training* Bahnar queries scored against this fixed Vietnamese test-candidate pool.

1. `Z(S)[q,c] = (S[q,c] - mean(S[q,:])) / (std(S[q,:]) + 1e-9)`; fusion `F=(Z(Dice)+Z(Char)+Z(E1))/3`.
2. Training score fusion `T` is computed using the same three row z-score transformations for all training queries against the same 780 candidates.
3. `density_k(c) = mean_top_k(T[:,c])` and `corrected_k(q,c) = 2F(q,c)-density_k(c)`.
4. Inverted Softmax `IS_beta(q,c) = beta*F(q,c) - logsumexp(beta*T[:,c])`.
5. **Strict ranking:** designated reference must score higher than every alternative. Ties count against the system. All matches and rankings use candidate IDs, not matrix diagonals.
6. R@1, R@5, R@10 and full-pool MRR are computed for the identical 781×780 held-out evaluation.
7. Two-sided exact paired McNemar is computed from baseline-only and correction-only hits, treating queries as the observational units. There are within-family correlations, so query-level p-values do not substitute for family-cluster uncertainty.

The original primary k=3 was fixed before opening the BA48 test; retrospective k-grid, Inverted Softmax and Sinkhorn selections were investigated after test access. Validation selecting k=10 and β=2 **after** the test had been opened does not produce a new independent experiment.

See `results/recomputed_test_metrics.json` for verified per-family and per-parameter score-level outcomes.
