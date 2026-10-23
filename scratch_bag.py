import time, numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
SEED=260006
data=fetch_olivetti_faces(shuffle=False); X=data.data.astype(np.float32); y=data.target.astype(int)
def split():
    rng=np.random.default_rng(SEED); tr=[];va=[];te=[]
    for c in range(40):
        idx=rng.permutation(np.flatnonzero(y==c)); tr.extend(idx[:6]);va.extend(idx[6:8]);te.extend(idx[8:10])
    return np.array(tr),np.array(va),np.array(te)
tr,va,te=split(); Xtr,ytr=X[tr],y[tr]; Xva,yva=X[va],y[va]; Xte,yte=X[te],y[te]
classes=np.unique(y); K=len(classes); n=len(tr)
rng=np.random.default_rng([SEED,11]); ids=np.stack([rng.integers(0,n,n) for _ in range(200)]); seeds=rng.integers(0,2**31-1,size=200)
def fit(i):
    m=DecisionTreeClassifier(max_depth=None,min_samples_leaf=1,random_state=int(seeds[i])).fit(Xtr[ids[i]],ytr[ids[i]]); return m
t0=time.time(); trees=Parallel(n_jobs=10,prefer='processes')(delayed(fit)(i) for i in range(200)); print('fit200',time.time()-t0)
Ptr=np.stack([t.predict(Xtr) for t in trees]); Pva=np.stack([t.predict(Xva) for t in trees]); Pte=np.stack([t.predict(Xte) for t in trees])
def vote(P,B):
    V=np.stack([(P[:B]==c).sum(0) for c in classes],1); return classes[V.argmax(1)], V/B
for B in [1,5,10,25,50,100,200]:
    ptr,_=vote(Ptr,B); pva,_=vote(Pva,B); pte,_=vote(Pte,B)
    print(B, 'train',np.mean(ptr==ytr),'val',np.mean(pva==yva),'test',np.mean(pte==yte),'distinct',np.mean([len(np.unique(ids[i])) for i in range(B)]))
# disagreement and margin validation
for B in [1,5,10,25,50,100,200]:
    p=Pva[:B]
    if B>1:
        dis=np.mean([np.mean(p[i]!=p[j]) for i in range(B) for j in range(i+1,B)])
    else: dis=0
    V=np.stack([(p==c).sum(0) for c in classes],1)/B
    s=np.sort(V,1); margin=np.mean(s[:,-1]-s[:,-2])
    print('B',B,'dis',round(dis,4),'margin',round(margin,4))
# oob
for B in [1,5,10,25,50,100,200]:
    V=np.zeros((n,K)); cnt=np.zeros(n)
    for i in range(B):
        oob=np.ones(n,bool); oob[np.unique(ids[i])]=False
        pp=trees[i].predict(Xtr[oob])
        for row,p in zip(np.flatnonzero(oob),pp): V[row,p]+=1
        cnt[oob]+=1
    val=cnt>0; pred=classes[V[val].argmax(1)]
    print('B',B,'oob',np.mean(pred==ytr[val]),'cov',val.mean())
# Q1 cases
res={'corr':[],'bothwrong':[]}
pva,V=vote(Pva,200); pte,Vte=vote(Pte,200); ptr,_=vote(Ptr,200)
t=DecisionTreeClassifier(max_depth=None,min_samples_leaf=1,random_state=SEED).fit(Xtr,ytr); tp=t.predict(Xte)
for i in range(len(te)):
    if tp[i]!=yte[i] and pte[i]==yte[i]: res['corr'].append((i,tp[i],pte[i],Vte[i].max()))
    if tp[i]!=yte[i] and pte[i]!=yte[i]: res['bothwrong'].append((i,tp[i],pte[i],Vte[i].max()))
print('corr',res['corr']); print('bothwrong',res['bothwrong'][:20])
