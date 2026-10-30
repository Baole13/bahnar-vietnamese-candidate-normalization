# Bahnar–Vietnamese Sentence Retrieval: Candidate Normalization and Evaluation

[![Reproduce E11 results](https://github.com/Baole13/bahnar-vietnamese-candidate-normalization/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Baole13/bahnar-vietnamese-candidate-normalization/actions/workflows/reproduce.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility: Score-Level](https://img.shields.io/badge/reproducibility-score--level-success.svg)](#quickstart)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Score-level research artifacts and evaluation scripts for **train-derived vs. transductive candidate normalization** in low-resource Bahnar → Vietnamese sentence retrieval. The repository accompanies our paper prepared for **ACIIDS 2027** (Springer LNCS/LNAI style).

> [!NOTE]
> Due to source copyright restrictions, this repository does **not** contain raw newspaper text or unredacted corpus PDFs. All reproducible scoring operations run entirely on frozen, pseudonymized score matrices (`*.npz`) without requiring GPU or private credentials.

---

## 📌 Research Question & Key Findings

With a fixed lexical–neural fusion scorer and a locked Vietnamese candidate index:
1. What happens when candidate density normalization is estimated strictly from **training queries** (CSLS-style $k=3$) versus relying transductively on the **unlabeled test-query batch** (Sinkhorn)?
2. How robust is inference when the incoming query batch is subsampled to smaller sizes?
3. What is the empirical impact of silver-reference alignment on evaluation validity?

### Summary of Results (BA48 Held-out Test: 780 queries, 780 candidates, 1-to-1 pool)

Strict silver-reference retrieval: any tie with the designated reference candidate counts as incorrect.

| Method | Role / Selection Status | Correct@1 | Strict R@1 | Paired vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| Equal-weight Fusion (Dice + Char TF–IDF + E1) | Fixed Baseline | 738/780 | 94.62% | — |
| **Train-query Density Correction ($k=3$)** | **Pre-specified Primary Confirmatory** | **748/780** | **95.90%** | +14 / -4 (McNemar $p=0.0309$) |
| Train-query Density Correction ($k=5$) | Declared Sensitivity | 749/780 | 96.03% | +13 / -2 (McNemar $p=0.0074$) |
| Train-query Density Correction ($k=10$) | Retrospective Validation-Selected | 751/780 | 96.28% | +14 / -1 (McNemar $p=0.0010$) |
| **Train-query Inverted Softmax ($\beta=2$)** | Retrospective Validation-Selected | **758/780** | **97.18%** | +20 / -0 (McNemar $p=2.0\times 10^{-6}$) |
| Transductive Sinkhorn ($\beta=5$, full batch) | Post-hoc Exploration | 769/780 | 98.59% | Sensitive to batch size (94.46% at 10% queries) |
| Transductive Sinkhorn ($\beta=2$, full batch) | Post-hoc Exploration | 764/780 | 97.95% | Retains 95.53% at 10% queries |

> [!IMPORTANT]
> **Evidence Status Distinction:**
> - **Confirmatory Primary ($k=3$):** Pre-specified protocol before test evaluation; 95% bootstrap confidence interval $[+0.12, +2.45]$ percentage points (10 families improve, 8 unchanged, 2 decline).
> - **Retrospective Validation-Selected ($k=10, \beta=2$):** Selected post hoc using the independent 768-query validation set ($k=10 \to 737/768$, $\beta=2 \to 736/768$). While fully reproducible from frozen validation matrices, its test result is an exploratory confirmation, not an independent blinded second study.
> - **Sinkhorn Balancing:** Exhibits acute performance degradation under subsampled query batches (falling from 98.59% full-batch to 94.46% at 10% query coverage).

---

## 🗺️ Paper Table-to-Code Mapping

Every empirical table and figure reported in the paper is linked to its script and output data:

| Paper Element | Description | Reproduction Command | Primary Output File |
| :--- | :--- | :--- | :--- |
| **Table 1** | Data splits & candidate pool (780 queries $\times$ 780 candidates) | *Inspect metadata* | [`docs/UPSTREAM_SOURCES.md`](docs/UPSTREAM_SOURCES.md), [`results/retained_query_manifest.csv`](results/retained_query_manifest.csv) |
| **Table 2** | Primary test retrieval (Fusion 738, $k=3 \to 748$) | `python scripts/evaluate_scores.py --check` | [`results/recomputed_test_metrics.json`](results/recomputed_test_metrics.json), [`results/method_results.csv`](results/method_results.csv) |
| **Table 3** | Temperature grid scan ($\beta \in \{0.5, 1, 2, 5, 10\}$) | `python scripts/evaluate_scores.py` | [`results/extra_k_beta_table.csv`](results/extra_k_beta_table.csv) |
| **Table 4 / Fig 1** | Batch subsampling stress test & Sinkhorn balancing | `python scripts/reproduce_sinkhorn_and_subsampling.py --check` | [`results/batch_stress_200.csv`](results/batch_stress_200.csv), [`results/query_batch_sensitivity_780.png`](results/query_batch_sensitivity_780.png) |
| **Validation Selection** | End-to-end parameter search ($k=10, \beta=2$) | `python scripts/val_select.py` | [`results/val_selection_results.json`](results/val_selection_results.json) |

---

## 🚀 Quickstart (Clean Clone Reproduction)

Reproducing all core score-level results requires only Python 3.10+ and standard numerical libraries:

```bash
# 1. Clone repository
git clone https://github.com/Baole13/bahnar-vietnamese-candidate-normalization.git
cd bahnar-vietnamese-candidate-normalization

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Reproduce frozen held-out test rankings (780x780 pool)
python scripts/evaluate_scores.py --check

# 4. Reproduce Sinkhorn balancing and subsampling stress test
python scripts/reproduce_sinkhorn_and_subsampling.py --check

# 5. Reproduce validation selection end-to-end
python scripts/val_select.py

# 6. Run automated test suite & verify cryptographic SHA-256 integrity
python -m unittest discover -s tests -v
python scripts/verify_sha256.py
```

Expected output for step 3:
```text
PASS: 738 baseline / 748 primary k3 / 751 k10 / 758 Inverted Softmax beta2; 20 families
```

Expected output for step 4:
```text
PASS: Sinkhorn rowcol beta=5 -> 769/780, beta=2 -> 764/780 on 1-to-1 pool
```

---

## 📁 Repository Layout

```text
bahnar-vietnamese-candidate-normalization/
├── .github/workflows/         # Automated GitHub Actions CI workflow
├── data/
│   ├── scores/                # Frozen NPZ matrices (TEST_scores, train_TESTpool, VAL_scores, train_VALpool)
│   └── metadata/              # Hash-derived query/candidate IDs and document families
├── scripts/
│   ├── evaluate_scores.py     # Main score evaluation and parameter grid scanning
│   ├── reproduce_sinkhorn_and_subsampling.py # Reproduces Sinkhorn and query-batch subsampling
│   ├── val_select.py          # End-to-end validation selection from NPZ matrices
│   ├── verify_sha256.py       # Independent SHA-256 checksum verifier
│   └── optional_with_source_text/ # Extended scripts requiring raw licensed corpus
├── tests/
│   └── test_reproduction.py   # Regression and strict-tie unit tests (780 pool)
├── results/                   # Recomputed test metrics, validation summaries, per-query flags
├── docs/
│   ├── DATA_CARD.md           # Dataset card: size, silver pipeline, and limitations
│   ├── EVALUATION_PROTOCOL.md # Formal evaluation definitions and tie-breaking rules
│   ├── REPRODUCIBILITY.md     # Reproducibility scope, information boundaries, and limits
│   ├── UPSTREAM_SOURCES.md    # Archive provenance and split construction details
│   └── DATA_PROVENANCE_AND_RIGHTS.md # Corpus copyright notice and ethics statement
├── requirements.txt           # Minimum Python dependencies (numpy, scipy, scikit-learn)
├── CITATION.cff               # Machine-readable citation metadata
├── LICENSE                    # Software source code license (MIT License)
└── SHA256SUMS                 # Cryptographic verification checksums for release files
```

---

## ⚖️ License & Provenance Notice

- **Software Code:** The evaluation and normalization code is licensed under the [MIT License](LICENSE) (Copyright (c) 2026 E11 Research Contributors).
- **Research Scope:** Code licenses do **not** grant redistribution rights to original news articles, corpus PDFs, or proprietary model checkpoints. See [Data Provenance & Rights](docs/DATA_PROVENANCE_AND_RIGHTS.md).

---

## 📜 Citation

If you use these artifacts or findings in your work, please cite:

```bibtex
@inproceedings{le2027candidate,
  title={When Does Candidate Normalization Help? A Low-Resource Bahnar--Vietnamese Sentence Retrieval Study},
  author={Le, Bao and others},
  booktitle={Proceedings of the 19th Asian Conference on Intelligent Information and Database Systems (ACIIDS 2027)},
  series={Lecture Notes in Computer Science},
  publisher={Springer},
  year={2027}
}
```
