import time, numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
SEED=260006
data=fetch_olivetti_faces(shuffle=False); X=data.data.astype(np.float32); y=data.target.astype(int)
def split():
    rng=np.random.default_rng(SEED); tr=[];va=[];te=[]
    for c in range(40):
        idx=rng.permutation(np.flatnonzero(y==c)); tr.extend(idx[:6]);va.extend(idx[6:8]);te.extend(idx[8:10])
    return np.array(tr),np.array(va),np.array(te)
tr,va,te=split(); Xtr,ytr=X[tr],y[tr]; Xva=X[va]; Xte=X[te]; classes=np.unique(y); K=40

def samme(depth, rounds, seed):
    rng=np.random.default_rng(seed); n=len(ytr); w=np.ones(n)/n; models=[]; alphas=[]; hist=[]
    t0=time.time()
    for t in range(rounds):
        m=DecisionTreeClassifier(max_depth=depth,random_state=int(rng.integers(0,2**31-1))).fit(Xtr,ytr,sample_weight=w)
        p=m.predict(Xtr); err=float(w[p!=ytr].sum())
        if err>=1-1/K: break
        a=float(np.log((1-err)/max(err,1e-12))+np.log(K-1)); models.append(m); alphas.append(a)
        w=w*np.exp(np.clip(a*(p!=ytr),-50,50)); w/=w.sum()
        hist.append((err,a,w.max(),np.sort(w)[-10:].sum()))
    def predict(Xq):
        S=np.zeros((len(Xq),K))
        for m,a in zip(models,alphas):
            pp=m.predict(Xq); S[np.arange(len(Xq)),pp]+=a
        return classes[S.argmax(1)], S/max(sum(alphas),1e-12)
    return len(models), np.mean(predict(Xtr)[0]==ytr), np.mean(predict(Xva)[0]==y[va]), np.mean(predict(Xte)[0]==y[te]), time.time()-t0, hist[-1] if hist else None
for d in [1,2,3,4]:
    for r in [25,100]:
        print(d,r, samme(d,r,SEED+100*d+r))
