#!/usr/bin/env python3
"""Check the packaged anonymized annotations and paired 18-query summary.

No claim is made that CSV itself establishes the reviewers' identities, independence,
consent, timing or original free-text responses. Those require the private forms.
"""
from pathlib import Path
from collections import defaultdict,Counter
from math import sqrt
from scipy.stats import binomtest
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def find_path(rel_a, rel_b):
    p_a = ROOT / rel_a
    return p_a if p_a.is_file() else ROOT / rel_b

rows = read_csv(find_path('data/human_eval/reviewer_labels_anonymized.csv', 'results/reviewer_labels_anonymized.csv'))
items=defaultdict(dict)
for r in rows:
    assert r['reviewer'] not in items[r['item']]
    items[r['item']][r['reviewer']]=r
assert len(rows)==176 and len(items)==88
assert all(set(x)=={'REV_A','REV_B'} for x in items.values())
identical=sum((v['REV_A']['label_A_or_equivalent'],v['REV_A']['label_B'])==(v['REV_B']['label_A_or_equivalent'],v['REV_B']['label_B']) for v in items.values())
confidence_differ=sum(v['REV_A']['confidence']!=v['REV_B']['confidence'] for v in items.values())
controls=[k for k in items if k.startswith('R')]
assert len(controls)==70
positive=sum(items[k]['REV_A']['label_A_or_equivalent']=='YES' for k in controls)
negative=sum(items[k]['REV_A']['label_A_or_equivalent']=='NO' for k in controls)
paired = read_csv(find_path('data/human_eval/HUMAN_VALIDATION_18_RESULT.csv', 'results/HUMAN_VALIDATION_18_RESULT.csv'))
assert len(paired)==18
strict=Counter(r['strict'] for r in paired);lenient=Counter(r['lenient'] for r in paired)
assert strict['fix']==13 and strict['reg']==5 and lenient['fix']==13 and lenient['reg']==3
s=lambda C:float(binomtest(C['fix'],C['fix']+C['reg'],.5).pvalue)
out={'labels_total':len(rows),'items':len(items),'labels_identical_on_items':identical,'confidence_differences':confidence_differ,'controls_yes':positive,'controls_no':negative,'strict':dict(strict),'strict_exact_mcnemar':s(strict),'lenient':dict(lenient),'lenient_exact_mcnemar':s(lenient),'silver_reference_usable_of_18':sum(r['agrees_with_silver_gold']=='True' for r in paired),'scope':'CSV aggregate audit; no proof of human provenance or independent collection'}
print(json.dumps(out,indent=2,ensure_ascii=False))
(ROOT/'results'/'human_audit_verified_from_export.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
