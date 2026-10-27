#!/usr/bin/env python3
"""Dice translation-table + character n-gram retrieval baseline (Bahnar -> Vietnamese).

Reproduces the 'strong baseline' used in the Bahnar A+B analysis.  No GPU, runs in about a minute.

Inputs (UTF-8 CSV with a header):
  --train   CSV with columns  ba,vi         (training sentence pairs; use ONLY training data)
  --queries CSV with columns  ba,vi         (row i of the pool is the gold candidate of query i;
                                             vi is used ONLY as the candidate text, never as a feature of the query)
Outputs in --out:
  scores_dice.npy, scores_char.npy, scores_fusion.npy   float32 [n_queries, n_candidates]
  metrics.json                                          R1/R5/R10/MRR10 for each system
Optional --mask-digits removes digit tokens from the Bahnar queries only (candidates unchanged).

Key design choices (each one matters; see the ablation in the analysis):
  * canonical tokens: lowercase, NFC, map ƀ->b, č->ch, ñ->nh, đ->d, strip tone marks and apostrophes
  * digit tokens are KEPT (numbers co-occur across languages and carry a lot of signal)
  * Dice association on sentence-level co-occurrence, min co-occurrence 2, top-5 translations per Bahnar token, L1-normalised
  * IDF-weighted bag of words on both sides; cosine between the projected query and each candidate
  * char n-gram (3-5, char_wb) TF-IDF cosine on the canonical text; fusion = z-score(dice) + z-score(char)
"""
import argparse, collections, csv, json, os, re, unicodedata
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def canon(s):
    s = unicodedata.normalize('NFC', s).lower().replace('ƀ', 'b').replace('č', 'ch').replace('ñ', 'nh').replace('đ', 'd')
    s = re.sub(r'[̀-ͯ]', '', unicodedata.normalize('NFD', s))
    return re.sub(r"[‘’ʼ'´`]", '', s)

def toks(s): return re.findall(r'[a-z0-9]+', canon(s))

def read_pairs(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return [(r['ba'], r['vi']) for r in csv.DictReader(f)]

def build_table(train, topk=5, min_cooc=2, min_df=2):
    trb = [toks(a) for a, _ in train]; trv = [toks(b) for _, b in train]
    def vocab(docs):
        c = collections.Counter(w for d in docs for w in set(d))
        return {w: i for i, (w, k) in enumerate(c.most_common()) if k >= min_df}
    vb, vv = vocab(trb), vocab(trv)
    def mat(docs, voc):
        r, c = [], []
        for i, d in enumerate(docs):
            for w in set(d):
                j = voc.get(w)
                if j is not None: r.append(i); c.append(j)
        return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(docs), len(voc)))
    Xb, Xv = mat(trb, vb), mat(trv, vv)
    C = (Xb.T @ Xv).tocoo()
    dfb = np.asarray(Xb.sum(0)).ravel(); dfv = np.asarray(Xv.sum(0)).ravel()
    ok = C.data >= min_cooc
    M = sp.csr_matrix((2 * C.data[ok] / (dfb[C.row[ok]] + dfv[C.col[ok]]), (C.row[ok], C.col[ok])), shape=C.shape)
    rows, cols, vals = [], [], []
    for i in range(M.shape[0]):
        s, e = M.indptr[i], M.indptr[i + 1]
        if e == s: continue
        d, ix = M.data[s:e], M.indices[s:e]
        if e - s > topk:
            sel = np.argpartition(-d, topk)[:topk]; d, ix = d[sel], ix[sel]
        rows += [i] * len(d); cols += list(ix); vals += list(d)
    T = normalize(sp.csr_matrix((vals, (rows, cols)), shape=M.shape), norm='l1')
    idb = np.log((1 + Xb.shape[0]) / (1 + dfb)) + 1; idv = np.log((1 + Xv.shape[0]) / (1 + dfv)) + 1
    return vb, vv, T, idb, idv

def vec(docs, voc, idf):
    r, c, v = [], [], []
    for i, s in enumerate(docs):
        for w in set(toks(s)):
            j = voc.get(w)
            if j is not None: r.append(i); c.append(j); v.append(idf[j])
    return sp.csr_matrix((v, (r, c)), shape=(len(docs), len(voc)), dtype=np.float32)

def zs(S): return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)

def metrics(S):
    n = S.shape[0]; rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1) + 1
    return {'n': int(n), 'R1': float((rank == 1).mean()), 'R5': float((rank <= 5).mean()),
            'R10': float((rank <= 10).mean()), 'MRR10': float(np.where(rank <= 10, 1 / rank, 0).mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True); ap.add_argument('--queries', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--mask-digits', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    train = read_pairs(a.train); q = read_pairs(a.queries)
    qb = [x for x, _ in q]; qv = [y for _, y in q]
    if a.mask_digits: qb = [' '.join(t for t in s.split() if not re.search(r'\d', t)) or s for s in qb]
    vb, vv, T, idb, idv = build_table(train)
    S_dice = (normalize(vec(qb, vb, idb) @ T) @ normalize(vec(qv, vv, idv)).T).toarray().astype(np.float32)
    cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
    cv.fit([canon(x) for x, _ in train] + [canon(y) for _, y in train] + [canon(x) for x in qb] + [canon(x) for x in qv])
    S_char = (cv.transform([canon(x) for x in qb]) @ cv.transform([canon(x) for x in qv]).T).toarray().astype(np.float32)
    S_fus = (zs(S_dice) + zs(S_char)).astype(np.float32)
    for name, S in [('dice', S_dice), ('char', S_char), ('fusion', S_fus)]:
        np.save(os.path.join(a.out, f'scores_{name}.npy'), S)
    res = {k: metrics(S) for k, S in [('dice', S_dice), ('char', S_char), ('fusion', S_fus)]}
    res['config'] = {'train_pairs': len(train), 'mask_digits': a.mask_digits}
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == '__main__': main()
