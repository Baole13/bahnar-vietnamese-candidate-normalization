# Reproducibility statement

## What can be reproduced without restricted source material?

The score matrices contain 3 channels each (`dice`, `char`, `neural_E1`), pseudonymous `query_ids`/`candidate_ids`, and the training-reference candidate order. A `gold_candidate_id` is provided for the held-out queries in `data/metadata/test_reference_ids.csv` (excluding one duplicate query `Q-e96d59ce6d0cbaf7` sharing an identical candidate with `Q-6f9096449df40a3e`, yielding a strict 1-to-1 matching pool of 780 queries and 780 candidates).

`python scripts/evaluate_scores.py --check` and `python scripts/val_select.py` verify:

- 738/780 correct for frozen fusion; 748/780 for train-reference k=3.
- 14 k3 fixes and 4 regressions relative to baseline (exact McNemar $p=0.0309$).
- 751/780 for k=10; 758/780 for Inverted Softmax β=2 (20 fixes, 0 regressions).
- Validation parameter selection reproduced end-to-end: baseline 725/768, peak k=10 (737/768), peak β=2 (736/768) via `python scripts/val_select.py`.
- Strict rank: `rank = count(score(candidate) >= score(designated_gold))`, including the gold itself. Thus a tied candidate is penalized.
- Correct top-1, top-5, top-10, full-pool MRR and the 20 document-family disaggregations.
- Transductive Sinkhorn and query-batch subsampling reproduced via `python scripts/reproduce_sinkhorn_and_subsampling.py --check` (Sinkhorn $\beta=5 \to 769/780$, $\beta=2 \to 764/780$; falling to 94.46% at 10% query coverage).

## What is *not* reproduced by this release?

- Fine-tuning the E1 encoder and re-extracting its embedding scores from raw text.
- Historical 55,324 pair source rebuilding, PDF font decoding, segmentation and sentence alignment.
- Independent validation of reviewer identity, expertise, consent or the original blind protocol from non-identifiable CSV counts.

The scripts under `scripts/optional_with_source_text/` have additional requirements; these inputs are not distributed here. Their presence is not a claim that score-level artifacts are sufficient for full end-to-end reproduction.

## Scientific status labels

| Result | Status |
|---|---|
| 738 baseline vs 748, k=3 | Frozen primary |
| k=5 (749/780) | Declared sensitivity |
| k=10 (751/780) via validation selection | Retrospective validation-selected; opened-test study |
| Inverted Softmax β=2 (758/780) via validation selection | Retrospective validation-selected; opened-test study |
| All other test k/β grid configurations | Post-hoc descriptive |
| Sinkhorn score (769/780 at $\beta=5$) and batch perturbations | Post-hoc exploratory, sensitive to query batch coverage |
| Human strict / lenient on 18 changing queries | Post-hoc aggregate spot check, not full gold |
