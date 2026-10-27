#!/usr/bin/env python3
"""Frozen E11 evaluation, k=3 primary; k=5 sensitivity; k=10 post-hoc.

DO NOT RUN ON SEALED TEST until all three test-channel matrices and the
matching three-channel train-query × TEST-candidate matrix are finalized.
NPZ input per split requires keys: candidate_ids, query_ids/train_query_ids,
dice, char, neural_E1. This program does NOT calculate embeddings or train.

Validation sanity example (allowed; must reproduce 725 and 734):
python locked_evaluator_UNRUN_ON_TEST.py --train-scores train_validpool.npz \\
  --evaluation-scores valid_scores.npz --eval-kind validation \\
  --queries-validation validation_queries_E09_768.jsonl --output ./sanity

Sealed-test execution (when separately authorized):
python locked_evaluator_UNRUN_ON_TEST.py --train-scores train_TESTpool.npz \\
  --evaluation-scores TEST_scores.npz --eval-kind sealed-test \\
  --evaluation-zip Bahnar_E2_v03_05_evaluation_docs.zip \\
  --leakage-flags ./audit/flagged_query_ids_only.csv --output ./heldout_results
"""
import argparse,csv,json,zipfile
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def rowz(A):
    A=np.asarray(A,np.float32)
    return (A-A.mean(axis=1,keepdims=True))/(A.std(axis=1,keepdims=True)+np.float32(1e-9))

def get_fusion(filename, train):
    with np.load(filename,allow_pickle=False) as npz:
        ids=npz['train_query_ids' if train else 'query_ids'].astype(str).tolist()
        cols=npz['candidate_ids'].astype(str).tolist()
        ss=[npz[k].astype(np.float32) for k in ('dice','char','neural_E1')]
    if len(set(cols))!=len(cols) or len(set(ids))!=len(ids):raise ValueError('Duplicate IDs')
    if any(a.shape!=(len(ids),len(cols)) for a in ss):raise ValueError('Score shapes inconsistent')
    if not all(np.isfinite(a).all() for a in ss):raise ValueError('Non-finite scores')
    F=(rowz(ss[0])+rowz(ss[1])+rowz(ss[2]))/np.float32(3)
    return ids,cols,F

def strict_correct(scores,gold_idx):
    gold=scores[np.arange(scores.shape[0]),gold_idx]
    # Every tie with the declared gold is counted against the system.
    return (scores>=gold[:,None]).sum(axis=1)==1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-scores',required=True)
    ap.add_argument('--evaluation-scores',required=True)
    ap.add_argument('--eval-kind',required=True,choices=['validation','sealed-test'])
    ap.add_argument('--evaluation-zip')
    ap.add_argument('--queries-validation')
    ap.add_argument('--leakage-flags')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    trainids,traincols,ft=get_fusion(args.train_scores,True)
    qids,cols,fe=get_fusion(args.evaluation_scores,False)
    if traincols!=cols:raise ValueError('TRAIN and EVAL candidate ID/order not identical')
    if set(trainids)&set(qids):raise ValueError('TRAIN/EVAL query ID overlap')
    if args.eval_kind=='validation':
        if not args.queries_validation:ap.error('--queries-validation required')
        with open(args.queries_validation,encoding='utf8') as f:querymeta=[json.loads(l) for l in f if l.strip()]
    else:
        if not args.evaluation_zip:ap.error('--evaluation-zip required')
        with zipfile.ZipFile(args.evaluation_zip) as z:
            with z.open('Bahnar_E2_v03/data/BA48_test_retrieval_queries.jsonl') as f:
                querymeta=[json.loads(l) for l in f if l.strip()]
    look={row['query_id']:row for row in querymeta}
    if set(qids)!=set(look):raise ValueError('Query IDs mismatch with dataset')
    colix={x:i for i,x in enumerate(cols)}
    gold_ids={q:look[q]['valid_candidate_ids'] for q in qids}
    if any(len(v)!=1 or v[0] not in colix for v in gold_ids.values()):raise ValueError('Gold mapping missing/ambiguous')
    skip=set()
    if args.leakage_flags:
        with open(args.leakage_flags,newline='',encoding='utf8') as f:
            skip={r['query_id'] for r in csv.DictReader(f)}
    use_idx=np.array([i for i,q in enumerate(qids) if q not in skip],dtype=int)
    if not len(use_idx):raise ValueError('No evaluation queries left')
    gold_idx=np.array([colix[gold_ids[q][0]] for q in qids],dtype=int)
    fam=np.array([look[q]['doc_family'] for q in qids],dtype=str)
    base=strict_correct(fe,gold_idx)
    output={'status':'test' if args.eval_kind=='sealed-test' else 'validation_sanity',
        'n_full':len(qids),'n_primary':len(use_idx),'n_family':len(set(fam[use_idx])),
        'n_candidates':len(cols),'primary_metric':'strict_silver_R@1; ties incorrect',
        'baseline_correct':int(base[use_idx].sum()),'baseline_R1':float(base[use_idx].mean())}
    variants=[]
    rows=[{'query_id':q,'family':fam[i], 'excluded':int(q in skip), 'baseline_correct':int(base[i])} for i,q in enumerate(qids)]
    for k,tag in [(3,'primary'),(5,'sensitivity'),(10,'posthoc_exploratory')]:
        if ft.shape[0]<k:raise ValueError('Too few train queries')
        density=np.partition(ft,ft.shape[0]-k,axis=0)[-k:,:].mean(axis=0)
        corrected=strict_correct(2*fe-density[None,:],gold_idx)
        b=base[use_idx];c=corrected[use_idx]
        fixes=int((~b & c).sum()); breaks=int((b & ~c).sum())
        p=float(binomtest(min(fixes,breaks),n=fixes+breaks,p=.5).pvalue) if fixes+breaks else 1.
        rng=np.random.default_rng(20261022)
        names=np.unique(fam[use_idx]);fi={g:use_idx[fam[use_idx]==g] for g in names}
        gains=[]
        for _ in range(5000):
            picked=rng.choice(names,size=len(names),replace=True)
            ix=np.concatenate([fi[g] for g in picked]);gains.append(float((corrected[ix].astype(int)-base[ix].astype(int)).mean()))
        ci=np.quantile(gains,[.025,.975])
        v={'k':k,'role':tag,'correct':int(c.sum()),'r1':float(c.mean()),'fixes':fixes,'breaks':breaks,
           'net_count':fixes-breaks,'delta_r1':float(c.mean()-b.mean()),'mcnemar_exact_two_sided_p':p,
           'family_bootstrap_delta_r1_95pct':[float(ci[0]),float(ci[1])],'bootstrap_reps':5000}
        variants.append(v)
        for i,r in enumerate(rows):r[f'k{k}_correct']=int(corrected[i])
    output['variants']=variants
    output['note']='k3 primary, k5 sensitivity, k10 posthoc; never choose k on test'
    d=Path(args.output);d.mkdir(parents=True,exist_ok=True)
    (d/'locked_evaluation_summary.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf8')
    fields=list(rows[0]);
    with (d/'paired_query_results.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    with (d/'per_family_summary.csv').open('w',encoding='utf8',newline='') as f:
        field=['family','n','baseline_correct','k3_correct','k5_correct','k10_correct','k3_net','k5_net','k10_net'];w=csv.DictWriter(f,field);w.writeheader()
        for g in sorted(set(fam[use_idx])):
            ix=use_idx[fam[use_idx]==g];m=[rows[i] for i in ix]
            w.writerow({'family':g,'n':len(ix),'baseline_correct':sum(r['baseline_correct'] for r in m),
                **{f'k{k}_correct':sum(r[f'k{k}_correct'] for r in m) for k in (3,5,10)},
                **{f'k{k}_net':sum(r[f'k{k}_correct']-r['baseline_correct'] for r in m) for k in (3,5,10)}})
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
