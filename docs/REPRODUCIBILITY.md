# Reproducibility statement

## What can be reproduced without restricted source material?

The two score matrices contain 3 channels each (`dice`, `char`, `neural_E1`), pseudonymous `query_ids`/`candidate_ids`, and the training-reference candidate order. A `gold_candidate_id` is provided for all 781 held-out queries in `data/metadata/test_reference_ids.csv` (from the original BA48 pairing, translated into ID form before text removal).

`python scripts/evaluate_scores.py --check` and `python scripts/val_select.py` verify:

- 739/781 correct for frozen fusion; 749/781 for train-reference k=3.
- 14 k3 fixes and 4 regressions relative to baseline.
- 752/781 for k=10; 759/781 for Inverted Softmax β=2 (20 fixes, 0 regressions).
- Validation parameter selection reproduced end-to-end: baseline 725/768, peak k=10 (737/768), peak β=2 (736/768) via `python scripts/val_select.py`.
- Strict rank: `rank = count(score(candidate) >= score(designated_gold))`, including the gold itself. Thus a tied candidate is penalized.
- Correct top-1, top-5, top-10, full-pool MRR and the 20 document-family disaggregations.

## What is *not* reproduced by this release?

- Fine-tuning the E1 encoder and re-extracting its embedding scores from raw text.
- Historical 55,324 pair source rebuilding, PDF font decoding, segmentation and sentence alignment.
- The historical transductive Sinkhorn report (retained as an exploratory, non-reproduced historical comparison; highly sensitive to query subsampling).
- Independent validation of reviewer identity, expertise, consent or the original blind protocol from non-identifiable CSV counts.

The scripts under `scripts/optional_with_source_text/` have additional requirements; these inputs are not distributed here. Their presence is not a claim that score-level artifacts are sufficient for full end-to-end reproduction.

## Scientific status labels

| Result | Status |
|---|---|
| 739 baseline vs 749, k=3 | Frozen primary |
| k=5 | Declared sensitivity |
| k=10 via validation selection | Retrospective validation-selected; opened-test study |
| Inverted Softmax β=2 via validation selection | Retrospective validation-selected; opened-test study |
| All other test k/β grid configurations | Post-hoc descriptive |
| Sinkhorn score and batch perturbations | Historical exploratory, not recreated here |
| Human strict / lenient on 18 changing queries | Post-hoc aggregate spot check, not full gold |
