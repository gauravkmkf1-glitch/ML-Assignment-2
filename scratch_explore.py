import time, itertools
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

SEED = 260006
data = fetch_olivetti_faces(shuffle=False)
X = data.data.astype(np.float32); y = data.target.astype(int)

def make_split(y):
    rng = np.random.default_rng(SEED)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        tr.extend(idx[:6]); va.extend(idx[6:8]); te.extend(idx[8:10])
    return np.array(tr), np.array(va), np.array(te)

tr, va, te = make_split(y)
print("split", len(tr), len(va), len(te), "seed", SEED)
for c in [0, 1, 2]:
    print(c, sorted(tr[y[tr]==c]), sorted(va[y[va]==c]), sorted(te[y[te]==c]))

def acc(m, X_, y_): return float(np.mean(m.predict(X_) == y_))
print("\nTREE GRID")
best = None
for d, leaf in itertools.product([2,4,8,16,None],[1,2,4]):
    t0=time.time()
    m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=SEED).fit(X[tr], y[tr])
    row=(d,leaf,acc(m,X[tr],y[tr]),acc(m,X[va],y[va]),acc(m,X[te],y[te]),m.get_n_leaves(),time.time()-t0)
    print(row)
    if best is None or row[3] > best[3]: best=row
print("BEST", best)

# bagging 100 trees at best
d, leaf = best[0], best[1]
rng = np.random.default_rng([SEED, 11])
n=len(tr); trees=[]; t0=time.time()
for b in range(100):
    ids = rng.integers(0,n,size=n)
    m=DecisionTreeClassifier(max_depth=d,min_samples_leaf=leaf,random_state=int(rng.integers(0,2**31-1))).fit(X[tr][ids],y[tr][ids])
    trees.append(m)
print("bag100 fit",time.time()-t0)
P=np.stack([t.predict(X[te]) for t in trees])
classes=np.unique(y)
votes=np.stack([(P==c).sum(0) for c in classes],1)
pred=classes[votes.argmax(1)]
print("bag100 test",np.mean(pred==y[te]),"train",np.mean(classes[np.stack([(np.stack([t.predict(X[tr]) for t in trees])==c).sum(0) for c in classes],1).argmax(1)]==y[tr]))

# SAMME 10 rounds depths
def samme(depth, rounds):
    rng=np.random.default_rng([SEED,22,depth])
    K=40; n=len(tr); w=np.ones(n)/n; models=[]; alphas=[]; hist=[]; t0=time.time()
    for t in range(rounds):
        m=DecisionTreeClassifier(max_depth=depth,random_state=int(rng.integers(0,2**31-1))).fit(X[tr],y[tr],sample_weight=w)
        p=m.predict(X[tr]); err=float(w[p!=y[tr]].sum())
        if err >= 1-1/K:
            print("  reject at",t,"err",err); break
        a=float(np.log((1-err)/max(err,1e-12))+np.log(K-1))
        models.append(m); alphas.append(a)
        w=w*np.exp(np.clip(a*(p!=y[tr]),-50,50)); w/=w.sum()
        hist.append((err,a,w.max(),np.sort(w)[-10:].sum()))
    def pred(X_):
        S=np.zeros((len(X_),K))
        for m,a in zip(models,alphas):
            pp=m.predict(X_)
            for j,c in enumerate(classes): S[:,j]+=a*(pp==c)
        return classes[S.argmax(1)], S/np.sum(alphas)
    return models,alphas,hist,pred,time.time()-t0

print("\nSAMME")
for d in [1,2,3,4]:
    models,alphas,hist,pred,dt=samme(d,10)
    print("depth",d,"actual",len(models),"train",np.mean(pred(X[tr])[0]==y[tr]),"val",np.mean(pred(X[va])[0]==y[va]),"test",np.mean(pred(X[te])[0]==y[te]),"time",dt)
    if hist: print("  last",hist[-1])
