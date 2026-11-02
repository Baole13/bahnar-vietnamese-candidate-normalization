# BA48 release manifest (text withheld)

BA48 is a silver-standard, automatically aligned Bahnar–Vietnamese sentence-pair corpus
(5,094 pairs from 170 issues of a trilingual provincial newspaper). This folder releases the
**structure** of the corpus, not the sentence text.

| File | Rows | Content |
| --- | --- | --- |
| `ba48_pairs_manifest.csv` | 5,094 | One row per pair: `ba48_row_id`, `split` (train / quarantined / validation / test), `evaluation_pool` (validation_768 / test_780 / excluded_duplicate), `query_id`, `vietnamese_candidate_id`, `doc_family`, `source_doc_id`, `pair_sha256_nfc_casefold_ws` |
| `ba48_source_pdfs.csv` | 170 | One row per source PDF: public URL, its SHA-256, document family, split, number of pairs, and verification flags |

Split counts: train 3,327, quarantined 218, validation 768, test 781 (780 in the one-to-one test pool; `BA48-001953` is excluded because it shares a Vietnamese reference with `BA48-000706`). 107 document families, each in exactly one split.

## What is not here
Bahnar and Vietnamese sentence text, the PDFs, and the original alignment/font-decoding code are not redistributed. A public PDF URL is not a redistribution licence, and the publisher's current notice requires written approval for copying. The sentence text will be released only with the publisher's written permission, within the scope that permission allows.

## Using the manifest
`pair_sha256_nfc_casefold_ws` is the SHA-256 of the pair text after Unicode NFC, case-folding and whitespace normalisation. Anyone who legitimately holds the source text can check that their copy matches. Hashes are pseudonymous, not anonymous.
`verified` columns in `ba48_source_pdfs.csv` mean what they say: `NO` marks masthead, publisher and copyright holder as not yet verified for that issue.

Labels are silver: no pair has been verified by a native speaker in full (see `docs/HUMAN_EVALUATION.md` for the targeted spot checks).

The manifest is covered by `SHA256SUMS`; run `python scripts/verify_sha256.py`.
