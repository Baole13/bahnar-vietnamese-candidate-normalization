#!/usr/bin/env python3
"""Reproduce retrospective *validation-selected* training-reference corrections.

No hard-coded machine paths. Original training/validation text and archived score matrices
are NOT redistributed by this kit; provide them locally via arguments.

Typical use:
 PYTHONHASHSEED=2 python scripts/val_select.py \
  --scorer /path/to/dice_char_baseline.py --train /path/to/train.csv \
  --val-scores /path/to/E09_original_aligned_768x768.npz \
  --val-queries /path/to/validation_queries.jsonl \
  --val-candidates /path/to/validation_candidates.jsonl \
  --train-queries /path/to/train_BA48_PAIRS_3327.jsonl \
  --neural-train-val /path/to/E11_neural_E1_train_3327_by_768.npz \
  --test-scores /path/to/TEST_scores.npz \
  --train-test-scores /path/to/train_TESTpool.npz \
  --test-queries /path/to/BA48_test_retrieval_queries.jsonl

The test was opened before this retrospective experiment was designed; its results
are NOT a new confirmatory study regardless of the validation-selection rule.
"""
from __future__ import annotations
import argparse,csv,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def read_jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def get_id(rec,options):
    for k in options:
        if k in rec:return rec[k]
    raise KeyError(f'Missing identifier; tried {options}')

def strict_ok(scores,gold):
    return (scores >= scores[np.arange(len(scores)),gold][:,None]).sum(axis=1)==1

def rz(a):
    a=np.asarray(a,dtype=np.float64)
    return (a-a.mean(axis=1,keepdims=True))/(a.std(axis=1,keepdims=True)+1e-9)

def density(F,k): return np.sort(F,axis=0)[-k:,:].mean(axis=0)

def inv(F,Ft,b): return b*F-logsumexp(b*Ft,axis=0)[None,:]

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ('scorer','train','val-scores','val-queries','val-candidates','train-queries','neural-train-val','test-scores','train-test-scores','test-queries'):
        ap.add_argument('--'+arg,required=True,type=Path)
    ap.add_argument('--output',default='val_selection_reproduced.json',type=Path)
    args=ap.parse_args()
    for k,p in vars(args).items():
        if k!='output' and not p.is_file():ap.error(f'{k} not found: {p}')
    spec=importlib.util.spec_from_file_location('dcb',args.scorer)
    dcb=importlib.util.module_from_spec(spec);spec.loader.exec_module(dcb)
    E09=np.load(args.val_scores,allow_pickle=False)
    qid=E09['query_ids'].astype(str).tolist();cid=E09['candidate_ids'].astype(str).tolist()
    vq={str(get_id(r,('query_id',))):r for r in read_jsonl(args.val_queries)}
    vc={str(get_id(r,('candidate_id',))):r for r in read_jsonl(args.val_candidates)}
    assert len(qid)==768 and len(cid)==768
    assert len(set(qid))==768 and len(set(cid))==768
    qba=[vq[q]['bahnar'] for q in qid]
    cvi=[vc[c]['vietnamese'] for c in cid]
    cix={c:i for i,c in enumerate(cid)}
    gold=np.array([cix[get_id(vq[q],('valid_candidate_ids',))[0]] for q in qid])
    hist=dcb.read_pairs(str(args.train))
    vb,vv,T,idb,idv=dcb.build_table(hist)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True,dtype=np.float32)
    char.fit([dcb.canon(x) for x,_ in hist]+[dcb.canon(y) for _,y in hist])
    def lexical(queries,candidates):
        D=(normalize(dcb.vec(queries,vb,idb)@T)@normalize(dcb.vec(candidates,vv,idv)).T).toarray().astype(np.float32)
        H=(char.transform([dcb.canon(x) for x in queries])@char.transform([dcb.canon(y) for y in candidates]).T).toarray().astype(np.float32)
        return D,H
    D,H=lexical(qba,cvi)
    refD=E09['dice'];refH=E09['char_inductive']
    errs=dict(dice_max_abs=float(np.max(np.abs(D-refD))),char_max_abs=float(np.max(np.abs(H-refH))))
    if not np.array_equal(D,refD) or not np.array_equal(H,refH):
        raise RuntimeError(f'Lexical scorer not identical to archived inductive scores: {errs}. Check PYTHONHASHSEED=2, sklearn and scorer version.')
    F=(rz(refD)+rz(refH)+rz(E09['neural_E1']))/3
    assert int(strict_ok(F,gold).sum())==725
    N=np.load(args.neural_train_val,allow_pickle=False)
    assert N['scores'].shape==(3327,768)
    assert N['candidate_ids'].astype(str).tolist()==cid, 'Candidate ID order mismatch'
    tq={str(get_id(r,('train_query_id','query_id'))):r for r in read_jsonl(args.train_queries)}
    train_ids=N['train_query_ids'].astype(str).tolist()
    TQ=[tq[q]['bahnar'] for q in train_ids]
    TD,TH=lexical(TQ,cvi)
    Ft=(rz(TD)+rz(TH)+rz(N['scores']))/3
    grid={}
    for k in range(1,11):grid[('k',k)]=strict_ok(2*F-density(Ft,k),gold)
    for b in (.5,1,2,5,10):grid[('beta',b)]=strict_ok(inv(F,Ft,b),gold)
    picked={}
    for family in ('k','beta'):
        arr=[x for x in grid if x[0]==family]
        picked[family]=sorted(arr,key=lambda a:(-int(grid[a].sum()),a[1]))[0]
    test=np.load(args.test_scores,allow_pickle=False);tref=np.load(args.train_test_scores,allow_pickle=False)
    assert np.array_equal(test['candidate_ids'],tref['candidate_ids'])
    tmeta={r['query_id']:r for r in read_jsonl(args.test_queries)}
    tqid=test['query_ids'].astype(str).tolist();tcid=test['candidate_ids'].astype(str).tolist()
    assert len(tqid)==781 and len(tcid)==780
    idx={c:i for i,c in enumerate(tcid)}
    tg=np.array([idx[tmeta[q]['valid_candidate_ids'][0]] for q in tqid])
    T=(rz(test['dice'])+rz(test['char'])+rz(test['neural_E1']))/3
    Tr=(rz(tref['dice'])+rz(tref['char'])+rz(tref['neural_E1']))/3
    base=strict_ok(T,tg);assert int(base.sum())==739
    outputs={}
    for family,(name,param) in picked.items():
        S=2*T-density(Tr,param) if family=='k' else inv(T,Tr,param)
        ok=strict_ok(S,tg)
        fixes=int((~base&ok).sum());breaks=int((base&~ok).sum())
        outputs[family]={'selected_param':param,'correct':int(ok.sum()),'fixes':fixes,'breaks':breaks,'mcnemar_exact':float(binomtest(fixes,fixes+breaks,.5).pvalue)}
    out={'status':'retrospective_validation_selection_after_test_access',
         'primary_frozen_unchanged':{'k':3,'test_correct':749},
         'validation_baseline':725,'lexical_check':errs,
         'validation_R1':{f'{name}{par}':int(mask.sum()) for (name,par),mask in grid.items()},
         'selected_validation_only':{k:v[1] for k,v in picked.items()},
         'test_retrospective':outputs}
    args.output.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
