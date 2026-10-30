# Data Card: BA48 Retrieval Benchmark

## 1. Dataset Overview
- **Name:** BA48 Bahnar--Vietnamese Parallel Sentence Retrieval Benchmark.
- **Language Pair:** Bahnar (ISO 639-3: `bdq`, low-resource) $\to$ Vietnamese (`vie`, high-resource).
- **Domain:** Journalistic text published in *Bao anh Gia Lai* (Bao Gia Lai archives).
- **Direction:** Retrieval of Vietnamese candidate sentences given Bahnar queries (evaluating against presumed translation direction: Vietnamese original $\to$ Bahnar translation).

---

## 2. Size & Partitions

| Split | Queries | Vietnamese Candidates | Document Families | Reference Status |
| :--- | :---: | :---: | :---: | :---: |
| **Training (BA48 Train)** | 3,327 | 3,327 | 69 | Silver (Automatic) |
| **Validation (E09)** | 768 | 768 | 18 | Silver (Automatic) |
| **Held-out Test (E11)** | 780 | 780 | 20 | Silver (Automatic) |

> After excluding one duplicate query (`Q-e96d59ce6d0cbaf7` / `BA48-001953`) that pointed to the same candidate as `Q-6f9096449df40a3e`, the held-out test split forms an exact 1-to-1 matching pool of 780 queries and 780 candidates.

---

## 3. Label Provenance & Silver Nature
- **Alignment:** Pairs were automatically extracted and aligned from bilingual PDF archives using OCR/legacy font recovery and automated bitext alignment tools.
- **Silver Label Constraint:** No row-level gold adjudication was performed by certified linguists across the full 5,094 pairs. Evaluation metrics measure alignment agreement against designated silver counterparts, not absolute semantic equivalence.
- **Human Spot-Check:** A targeted post-hoc audit by two independent Bahnar native speakers on 18 discordance items confirmed 17/18 silver pairings as usable (see `docs/HUMAN_EVALUATION.md`).

---

## 4. Release Format & Pseudonymization
- **Released Artifacts:** Frozen query-by-candidate score matrices (`TEST_scores.npz`, `train_TESTpool.npz`, `VAL_scores.npz`, `train_VALpool.npz`) containing float32 similarity values across 3 channels (`dice`, `char`, `neural_E1`).
- **Identifiers:** Pseudonymized hash IDs (`Q-...`, `TQ-...`, `VI-...`, `BAF-...`).
- **Withheld Artifacts:** Raw Vietnamese/Bahnar sentence texts, newspaper PDFs, proprietary neural encoder checkpoints, and unredacted reviewer forms are strictly withheld due to third-party copyright.

---

## 5. Limitations & Intended Use
- **Intended Use:** Purely for scientific reproducibility of candidate normalization rules (CSLS, Inverted Softmax, Sinkhorn) in cross-lingual sentence retrieval.
- **Out of Scope:** Not suitable for evaluating unconstrained open-world search, out-of-domain dialect generalization, or generative translation without additional resources.
