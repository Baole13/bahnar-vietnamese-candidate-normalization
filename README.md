# Bahnar–Vietnamese Sentence Retrieval: Candidate Normalization and Evaluation

[![Reproduce E11 results](https://github.com/Baole13/bahnar-vietnamese-candidate-normalization/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Baole13/bahnar-vietnamese-candidate-normalization/actions/workflows/reproduce.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility: Score-Level](https://img.shields.io/badge/reproducibility-score--level-success.svg)](#quickstart)

Score-level research artifacts and LaTeX publication sources for **train-derived vs. transductive candidate normalization** in low-resource Bahnar → Vietnamese sentence retrieval. The repository accompanies our submission prepared for **ACIIDS 2027** (Springer LNCS/LNAI style).

> [!NOTE]
> Due to source copyright restrictions, this repository does **not** contain raw newspaper text or unredacted corpus PDFs. All reproducible scoring operations run entirely on frozen, pseudonymized score matrices (`*.npz`) without requiring GPU or private credentials.

---

## 📌 Research Question & Key Findings

With a fixed lexical–neural fusion scorer and a locked Vietnamese candidate index:
1. What happens when candidate density normalization is estimated strictly from **training queries** (CSLS-style $k=3$) versus relying transductively on the **unlabeled test-query batch** (Sinkhorn)?
2. How robust is inference when the incoming query batch is subsampled to smaller sizes?
3. What is the empirical impact of silver-reference alignment on evaluation validity?

### Summary of Results (BA48 Held-out Test: 781 queries, 780 candidates)

Strict silver-reference retrieval: any tie with the designated reference candidate counts as incorrect.

| Method | Role / Selection Status | Correct@1 | Strict R@1 | Paired vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| Equal-weight Fusion (Dice + Char TF–IDF + E1) | Fixed Baseline | 739/781 | 94.62% | — |
| **Train-query Density Correction ($k=3$)** | **Pre-specified Primary Confirmatory** | **749/781** | **95.90%** | +14 / -4 (McNemar $p=0.0309$) |
| Train-query Density Correction ($k=10$) | Retrospective Validation-Selected | 752/781 | 96.29% | +14 / -1 (McNemar $p=0.0010$) |
| **Train-query Inverted Softmax ($\beta=2$)** | Retrospective Validation-Selected | **759/781** | **97.18%** | +20 / -0 (McNemar $p=1.9\times 10^{-6}$) |
| Transductive Sinkhorn ($\beta=5$, full batch) | Historical Post-hoc Report | 770/781 | 98.59% | Sensitive to query batch size |
| IBM Model 1 Symmetric (Nguyen et al. baseline) | Retrospective Within-Pool Baseline | 405/781 | 51.86% | Corrects 9 baseline residual errors |

> [!IMPORTANT]
> **Evidence Status Distinction:**
> - **Confirmatory Primary ($k=3$):** Pre-specified protocol before test evaluation; 95% bootstrap confidence interval $[+0.12, +2.44]$ percentage points.
> - **Retrospective Validation-Selected ($k=10, \beta=2$):** Selected post hoc using the independent 768-query validation set ($k=10 \to 737/768$, $\beta=2 \to 736/768$). While fully reproducible from frozen validation matrices, its test result is an exploratory confirmation, not an independent blinded second study.
> - **Sinkhorn Balancing:** Exhibits acute performance degradation under subsampled query batches (falling from 98.59% to 94.67% at 10% query subsampling).

---

## 🗺️ Paper Table-to-Code Mapping

Every numerical result in the manuscript corresponds to a dedicated reproduction script and verified output artifact:

| Paper Element | Description | Reproduction Command | Primary Output File |
| :--- | :--- | :--- | :--- |
| **Table 1** | Data composition & split partitions | *Inspect metadata* | [`docs/UPSTREAM_SOURCES.md`](docs/UPSTREAM_SOURCES.md) |
| **Table 2** | Primary test retrieval & IBM1 baseline | `python scripts/evaluate_scores.py --check` | [`results/recomputed_test_metrics.json`](results/recomputed_test_metrics.json) |
| **Table 3** | Temperature grid scan ($\beta \in \{0.5, 1, 2, 5, 10\}$) | `python scripts/evaluate_scores.py` | [`results/extra_k_beta_table.csv`](results/extra_k_beta_table.csv) |
| **Table 4** | Subsampling stress test (10% to 100% queries) | `python scripts/evaluate_scores.py` | [`results/extra_results.json`](results/extra_results.json) |
| **Table 5 / Sec 4b** | Native speaker audit (18 discordances, 70 controls) | `python scripts/analyze_human_validation.py` | [`results/human_audit_verified_from_export.json`](results/human_audit_verified_from_export.json) |
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

# 3. Reproduce frozen held-out test rankings
python scripts/evaluate_scores.py --check

# 4. Reproduce validation selection end-to-end
python scripts/val_select.py

# 5. Reproduce native-speaker evaluation audit
python scripts/analyze_human_validation.py

# 6. Run automated test suite & verify SHA-256 integrity
python -m unittest discover -s tests -v
python scripts/verify_sha256.py
```

Expected output for step 3:
```text
PASS: 739 baseline / 749 primary k3 / 752 k10 / 759 Inverted Softmax beta2; 20 families
```

---

## 📁 Repository Layout

```text
bahnar-vietnamese-candidate-normalization/
├── .github/workflows/         # Automated GitHub Actions CI workflow
├── data/
│   ├── scores/                # Frozen NPZ matrices (TEST_scores, train_TESTpool, VAL_scores, train_VALpool)
│   ├── metadata/              # Hash-derived query/candidate IDs and document families
│   └── human_eval/            # Anonymized reviewer labels and 18-query discordance records
├── scripts/
│   ├── evaluate_scores.py     # Main score evaluation and grid evaluation
│   ├── val_select.py          # End-to-end validation selection from NPZ matrices
│   ├── analyze_human_validation.py # Verification of reviewer paired labels
│   ├── verify_sha256.py       # Independent SHA-256 checksum verifier
│   └── optional_with_source_text/ # Extended scripts requiring raw licensed corpus
├── tests/
│   └── test_reproduction.py   # Regression and strict-tie unit tests
├── results/                   # Recomputed test metrics, validation summaries, per-query flags
├── docs/
│   ├── DATA_CARD.md           # Dataset card: size, silver pipeline, and limitations
│   ├── EVALUATION_PROTOCOL.md # Formal evaluation definitions and tie-breaking rules
│   ├── HUMAN_EVALUATION.md    # Targeted native-speaker review audit protocol
│   ├── REPRODUCIBILITY.md     # Reproducibility scope, information boundaries, and limits
│   ├── UPSTREAM_SOURCES.md    # Archive provenance and split construction details
│   └── DATA_PROVENANCE_AND_RIGHTS.md # Corpus copyright notice and ethics statement
├── paper/                     # Publication LaTeX source and compiled PDF manuscript (Springer LNCS)
├── requirements.txt           # Minimum Python dependencies (numpy, scipy, scikit-learn)
├── CITATION.cff               # Machine-readable citation metadata
├── LICENSE                    # Software source code license (Apache 2.0)
└── SHA256SUMS                 # Cryptographic verification checksums for release files
```

---

## ⚖️ License & Provenance Notice

- **Software Code:** The evaluation and normalization code is licensed under the [Apache 2.0 License](LICENSE).
- **Research Scope:** Code licenses do **not** grant redistribution rights to original news articles, corpus PDFs, or proprietary model checkpoints. See [Data Provenance & Rights](docs/DATA_PROVENANCE_AND_RIGHTS.md).
- **Human Evaluation Data:** [Aggregated anonymized evaluations](results/reviewer_labels_anonymized.csv) are published to confirm numerical agreement across 18 discordances and 70 controls without disclosing reviewer identities or raw private forms.

---

## 📜 Citation

If you use these artifacts or findings in your work, please cite:

```bibtex
@inproceedings{le2027candidate,
  title={When Does Candidate Normalization Help? A Low-Resource Bahnar--Vietnamese Sentence Retrieval Study},
  author={Le, Bao},
  booktitle={Proceedings of the 19th Asian Conference on Intelligent Information and Database Systems (ACIIDS 2027)},
  series={Lecture Notes in Computer Science},
  publisher={Springer},
  year={2027}
}
```
