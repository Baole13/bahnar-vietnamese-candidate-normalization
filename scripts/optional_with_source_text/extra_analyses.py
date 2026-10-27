#!/usr/bin/env python3
"""Extra analyses for the ACIIDS version (all use the frozen score matrices already archived).
Inputs: ../collab/in/{TEST_scores,train_TESTpool}.npz, BA48_test_retrieval_queries.jsonl
Outputs: extra_results.json + CSV tables. Seeds fixed."""
import argparse,json,sys,numpy as np,csv
from pathlib import Path
from scipy.stats import binomtest
from scipy.special import logsumexp
ap=argparse.ArgumentParser(description='Post-hoc analyses of archived E11 matrices; not confirmatory')
ap.add_argument('--input-dir',required=True,type=Path,help='Directory with TEST_scores.npz, train_TESTpool.npz, BA48_test_retrieval_queries.jsonl')
ap.add_argument('--output-dir',default=Path('.'),type=Path)
a=ap.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=True)
d=a.input_dir
T=np.load(d/'TEST_scores.npz');Tr=np.load(d/'train_TESTpool.npz')
qm={}
for l in open(d/'BA48_test_retrieval_queries.jsonl',encoding='utf-8'):
    if l.strip(): r=json.loads(l);qm[r['query_id']]=r
qid=T['query_ids'].astype(str).tolist();cid=T['candidate_ids'].astype(str).tolist()
gold=np.array([cid.index(qm[q]['valid_candidate_ids'][0]) for q in qid]);fam=np.array([qm[q]['doc_family'] for q in qid])
rz=lambda A:(A-A.mean(1,keepdims=True))/(A.std(1,keepdims=True)+1e-9);f64=lambda A:A.astype(np.float64)
F=(rz(f64(T['dice']))+rz(f64(T['char']))+rz(f64(T['neural_E1'])))/3
Ft=(rz(f64(Tr['dice']))+rz(f64(Tr['char']))+rz(f64(Tr['neural_E1'])))/3
n=len(qid);ar=np.arange(n)
def rank(S):  # strict rank: 1 + number of candidates scoring >= gold, excluding gold itself ( ties count against)
    g=S[ar,gold][:,None];return (S>=g).sum(1)
ok=lambda S:rank(S)==1
dk=lambda k:np.sort(Ft,0)[-k:].mean(0)
def metrics(S):
    r=rank(S);return dict(R1=int((r==1).sum()),R5=int((r<=5).sum()),R10=int((r<=10).sum()),MRR=float((1/r).mean()))
def inv(beta): return beta*F-logsumexp(beta*Ft,axis=0)[None,:]
base=ok(F);out={'n':n,'baseline':metrics(F)}
print('baseline',out['baseline'])
rows=[]
for k in range(1,11):
    S=2*F-dk(k);c=ok(S);fx=int((~base&c).sum());bk=int((base&~c).sum());p=binomtest(fx,fx+bk,.5).pvalue if fx+bk else 1.0
    m=metrics(S);rows.append(dict(method='train_k',param=k,fix=fx,brk=bk,p=round(p,4),**m))
for b in (0.5,1,2,5,10):
    S=inv(b);c=ok(S);fx=int((~base&c).sum());bk=int((base&~c).sum());p=binomtest(fx,fx+bk,.5).pvalue if fx+bk else 1.0
    m=metrics(S);rows.append(dict(method='train_inverted_softmax',param=b,fix=fx,brk=bk,p=round(p,4),**m))
for r in rows: print(r)
with open(a.output_dir/'extra_k_beta_table.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
out['grid']=rows
# per-family for k=3
c3=ok(2*F-dk(3));fr=[]
for f in np.unique(fam):
    m=fam==f;fr.append(dict(family=f,n=int(m.sum()),baseline=int(base[m].sum()),k3=int(c3[m].sum()),net=int(c3[m].sum()-base[m].sum())))
with open(a.output_dir/'extra_per_family_k3.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fr[0].keys());w.writeheader();w.writerows(fr)
print('families improved/same/worse',sum(x['net']>0 for x in fr),sum(x['net']==0 for x in fr),sum(x['net']<0 for x in fr))
out['per_family_net']=[x['net'] for x in fr]
# nested family-held-out selection of the hyperparameter
fams=np.unique(fam);idx={f:np.where(fam==f)[0] for f in fams}
cands={('k',k):ok(2*F-dk(k)) for k in range(1,11)}
cands.update({('beta',b):ok(inv(b)) for b in (0.5,1,2,5,10)})
rng=np.random.default_rng(20261026);M=4000
res={ 'baseline':[], 'k3_fixed':[], 'sel_k':[], 'sel_beta':[], 'sel_all':[]};chosen={'k':{}, 'beta':{}, 'all':{}}
for _ in range(M):
    perm=rng.permutation(fams);A=np.concatenate([idx[f] for f in perm[:10]]);B=np.concatenate([idx[f] for f in perm[10:]])
    for (S_,E_) in ((A,B),(B,A)):
        def pick(keys):
            sc=[(cands[kk][S_].mean(),-i) for i,kk in enumerate(keys)];return keys[int(np.argmax([s[0]+1e-9*s[1] for s in sc]))]
        kk=pick([x for x in cands if x[0]=='k']);bb=pick([x for x in cands if x[0]=='beta']);aa=pick(list(cands))
        res['baseline'].append(base[E_].mean());res['k3_fixed'].append(cands[('k',3)][E_].mean())
        res['sel_k'].append(cands[kk][E_].mean());res['sel_beta'].append(cands[bb][E_].mean());res['sel_all'].append(cands[aa][E_].mean())
        chosen['k'][kk[1]]=chosen['k'].get(kk[1],0)+1;chosen['beta'][bb[1]]=chosen['beta'].get(bb[1],0)+1;chosen['all'][str(aa)]=chosen['all'].get(str(aa),0)+1
summ={k:round(100*float(np.mean(v)),2) for k,v in res.items()}
print('nested family-held-out selection, mean R@1 on held-out half (%):',summ)
print('chosen k:',chosen['k']);print('chosen beta:',chosen['beta']);print('chosen overall:',chosen['all'])
out['nested']=dict(M=M,mean_R1_pct=summ,chosen=chosen,full_test_best_beta2=100*cands[('beta',2)].mean(),full_test_k3=100*c3.mean())
with (a.output_dir/'extra_results.json').open('w') as f:json.dump(out,f,indent=1,default=str)
