import time, numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
SEED=260006
data=fetch_olivetti_faces(shuffle=False); X=data.data.astype(np.float32); y=data.target.astype(int)
rng=np.random.default_rng(SEED); tr=[]
for c in range(40):
    idx=rng.permutation(np.flatnonzero(y==c)); tr.extend(idx[:6])
tr=np.array(tr); Xtr,ytr=X[tr],y[tr]
rng=np.random.default_rng([SEED,11]); ids=[rng.integers(0,len(tr),len(tr)) for _ in range(20)]; seeds=[int(rng.integers(0,2**31-1)) for _ in range(20)]
def fit(i):
    t=time.time(); m=DecisionTreeClassifier(max_depth=None,random_state=seeds[i]).fit(Xtr[ids[i]],ytr[ids[i]]); return m, time.time()-t
t0=time.time(); out=Parallel(n_jobs=10, prefer='processes', verbose=0)(delayed(fit)(i) for i in range(20)); print('parallel 20',time.time()-t0,'sum',sum(t for _,t in out))
t0=time.time(); out2=[fit(i) for i in range(20)]; print('serial 20',time.time()-t0)
print(all(np.array_equal(a.predict(Xtr),b.predict(Xtr)) for (a,_),(b,_) in zip(out,out2)))
