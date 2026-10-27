# Human semantic spot check: reporting boundaries

The v3.2 source artifact contains a **reported**, anonymized 88-item two-reviewer evaluation: 18 baseline–k3 disagreement cases, 60 unselected-by-discordance designated-reference controls, and 10 deliberately mismatched attention controls. The code-level audit reads 176 anonymized label rows and reports agreement on labels of 88/88 items, with confidence disagreements in 7 cases. Controls: 60 'YES' and 10 'NO'.

| Paired discordance criterion | Correction-only hits | Baseline-only hits | Two-sided exact McNemar |
|---|---:|---:|---:|
| Frozen silver designated reference | 14 | 4 | 0.0309 |
| Human strict (VALID only) | 13 | 5 | 0.0963 |
| Human lenient (VALID or PARTIAL) | 13 | 3 | 0.0213 |

The last two tests were **retrospective, targeted to model disagreement**, and have different definitions of semantic success. They do not provide new gold-accuracy estimates over the entire 781-query candidate pool. Selection and metric definitions must not be changed to optimize the p-value.

To verify numerical agreement without disclosing reviewer identities or raw private forms, anonymized de-identified labels are provided in `results/reviewer_labels_anonymized.csv` and `results/HUMAN_VALIDATION_18_RESULT.csv`, verified via `python scripts/analyze_human_validation.py`.
