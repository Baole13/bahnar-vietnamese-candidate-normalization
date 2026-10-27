# Anonymized Human Evaluation Records

This directory contains the de-identified, aggregated human spot-check records analyzed in **Section 4b** of the manuscript.

---

## 📂 Contents

1. [`reviewer_labels_anonymized.csv`](reviewer_labels_anonymized.csv):
   - **Total rows:** 176 rows (88 evaluation items $\times$ 2 independent native Bahnar reviewers: `REV_A`, `REV_B`).
   - **Items composition (88 items):**
     - **18 Discordance items (`D001` to `D018`):** Sentences where train-derived $k=3$ density normalization flipped the top-1 candidate relative to the uncorrected baseline.
     - **60 Designated Reference Controls (`R001` to `R060`):** Randomly selected pairs from the silver reference evaluation set.
     - **10 Negative Attention Controls (`R061` to `R070`):** Deliberately corrupted pairs with synthetic distractor sentences.
   - **Fields:**
     - `item`: Item identifier (`D...` or `R...`).
     - `reviewer`: Blind reviewer code (`REV_A` or `REV_B`).
     - `label_A_or_equivalent`: Rating for Candidate A (`VALID`, `PARTIAL`, `INVALID`, `YES`, `NO`).
     - `label_B`: Rating for Candidate B (for discordance pairs).
     - `confidence`: Reviewer self-assessed confidence level (`HIGH`, `MEDIUM`, `LOW`).

2. [`HUMAN_VALIDATION_18_RESULT.csv`](HUMAN_VALIDATION_18_RESULT.csv):
   - Paired consensus evaluation on the 18 discordant queries comparing the corrected system against baseline.
   - **Strict criterion (`VALID` only):** 13 fixes / 5 regressions (two-sided exact McNemar $p = 0.0963$).
   - **Lenient criterion (`VALID` or `PARTIAL`):** 13 fixes / 3 regressions / 2 ties (two-sided exact McNemar $p = 0.0213$).
   - Agreement with designated silver counterpart: **17 out of 18 queries** (94.4%).

---

## 🔍 Verification

Run the automated validation script:
```bash
python scripts/analyze_human_validation.py
```
Output:
```json
{
  "labels_total": 176,
  "items": 88,
  "labels_identical_on_items": 88,
  "confidence_differences": 7,
  "controls_yes": 60,
  "controls_no": 10,
  "strict": { "fix": 13, "reg": 5 },
  "strict_exact_mcnemar": 0.09625,
  "lenient": { "fix": 13, "tie": 2, "reg": 3 },
  "lenient_exact_mcnemar": 0.02127,
  "silver_reference_usable_of_18": 17
}
```

---

## ⚠️ Privacy & Boundary Note
Raw native-speaker questionnaires, identifiable consent forms, handwritten notes, and exact reviewer identities are strictly withheld in accordance with ethical human-subject evaluation standards.
