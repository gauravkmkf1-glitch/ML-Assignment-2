from __future__ import annotations
import json, time, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
ROLL = "P26AI0006"
SEED = int("".join(c for c in ROLL if c.isdigit()))
RNG = np.random.default_rng(SEED)
DEPTHS = [2, 4, 8, 16, None]
LEAFS = [1, 2, 4]
BAGS = [1, 5, 10, 25, 50, 100, 200]
BOOST_DEPTHS = [1, 2, 3, 4]
ROUNDS = [10, 25, 50, 100]
N_REPEATS = 30

def make_split(y):
    rng = np.random.default_rng(SEED)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        tr.extend(idx[:6]); va.extend(idx[6:8]); te.extend(idx[8:10])
    return np.array(tr), np.array(va), np.array(te)

def acc(model, X, y):
    return float(np.mean(model.predict(X) == y))

def leaf_stats(tree, X):
    leaves = tree.get_n_leaves()
    occupancy = tree.tree_.n_node_samples
    leaf_nodes = tree.tree_.children_left == tree.tree_.children_right
    counts = occupancy[leaf_nodes]
    return leaves, float(np.mean(counts)), float(np.mean(counts == 1))

def predict_vote(trees, X, classes):
    allp = np.stack([t.predict(X) for t in trees])
    votes = np.stack([(allp == c).sum(axis=0) for c in classes], axis=1)
    return classes[np.argmax(votes, axis=1)], votes / len(trees)

def pair_disagreement(trees, X):
    if len(trees) < 2: return 0.0
    preds = np.stack([t.predict(X) for t in trees])
    vals = [np.mean(preds[i] != preds[j]) for i in range(len(trees)) for j in range(i+1, len(trees))]
    return float(np.mean(vals))

def fit_bag(X, y, B, params, seed):
    rng = np.random.default_rng(seed)
    n = len(y); trees=[]; oob_votes = np.zeros((n, len(np.unique(y))), dtype=float)
    classes = np.unique(y); class_to_col = {c:i for i,c in enumerate(classes)}
    oob_count = np.zeros(n, dtype=int); distinct=[]
    for b in range(B):
        ids = rng.integers(0, n, size=n)
        distinct.append(len(np.unique(ids)))
        tree = DecisionTreeClassifier(**params, random_state=int(rng.integers(0, 2**31-1)))
        tree.fit(X[ids], y[ids]); trees.append(tree)
        oob = np.ones(n, dtype=bool); oob[np.unique(ids)] = False
        if np.any(oob):
            pp = tree.predict(X[oob])
            for row, pred in zip(np.flatnonzero(oob), pp):
                oob_votes[row, class_to_col[pred]] += 1
            oob_count[oob] += 1
    valid = oob_count > 0
    oob_pred = classes[np.argmax(oob_votes[valid], axis=1)] if np.any(valid) else np.array([])
    oob_acc = float(np.mean(oob_pred == y[valid])) if np.any(valid) else float("nan")
    return trees, oob_acc, float(np.mean(valid)), float(np.mean(distinct))

class SAMME:
    def __init__(self, depth, n_estimators, seed=SEED):
        self.depth=depth; self.n_estimators=n_estimators; self.seed=seed
        self.models=[]; self.alphas=[]; self.classes_=None; self.history=[]
    def fit(self, X, y):
        rng=np.random.default_rng(self.seed); self.classes_=np.unique(y); K=len(self.classes_)
        w=np.ones(len(y))/len(y)
        for t in range(self.n_estimators):
            model=DecisionTreeClassifier(max_depth=self.depth, random_state=int(rng.integers(0,2**31-1)))
            model.fit(X,y,sample_weight=w); pred=model.predict(X)
            err=float(w[pred != y].sum())
            if err >= 1 - 1/K:
                if not self.models: self.stopped_reason="weak learner not better than random"
                break
            alpha=float(np.log((1-err)/max(err,1e-12))+np.log(K-1))
            self.models.append(model); self.alphas.append(alpha)
            w *= np.exp(np.clip(alpha*(pred != y), -50, 50)); w /= w.sum()
            self.history.append({"round":len(self.models),"weighted_error":err,"alpha":alpha,
                "max_weight":float(w.max()),"top10_weight":float(np.sort(w)[-10:].sum())})
            if err <= 1e-12: break
        if not self.models: raise RuntimeError("SAMME stopped before accepting a weak learner. Try a different seed/data setup.")
        return self
    def scores(self, X):
        out=np.zeros((len(X),len(self.classes_)))
        for m,a in zip(self.models,self.alphas):
            pred=m.predict(X)
            for j,c in enumerate(self.classes_): out[:,j] += a*(pred==c)
        return out
    def predict(self, X): return self.classes_[np.argmax(self.scores(X),axis=1)]
    def proba_vote(self, X):
        s=self.scores(X); denom=np.sum(self.alphas)
        return s/denom if denom else np.zeros_like(s)

def main():
    t0=time.time()
    print(f"Student: Kumar Gaurav | Roll: {ROLL} | Seed: {SEED}")
    print("Loading Olivetti Faces (first run may download data)...")
    data=fetch_olivetti_faces(shuffle=False)
    X=data.data.astype(np.float32); y=data.target.astype(int)
    tr,va,te=make_split(y)
    np.savez(RES/"split.npz", train=tr, validation=va, test=te, seed=SEED)
    np.savez(RES/"dataset_meta.npz", targets=y, images=data.images)
    Xtr,Xva,Xte=X[tr],X[va],X[te]; ytr,yva,yte=y[tr],y[va],y[te]
    classes=np.unique(y)
    Xm=Xte.copy(); Xm[:,24*64+24:24*64+40] = Xm[:,24*64+24:24*64+40] # placeholder no-op
    Xm=data.images[te].copy(); Xm[:,24:40,24:40]=0; Xm=Xm.reshape(len(te),-1)

    # Q1(a): select only by validation accuracy; tie breaks in grid order.
    rows=[]; best=None
    for depth, leaf in itertools.product(DEPTHS, LEAFS):
        model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=SEED)
        model.fit(Xtr,ytr)
        ls,mo,sf=leaf_stats(model,Xtr)
        row={"depth":str(depth),"min_samples_leaf":leaf,"train_accuracy":acc(model,Xtr,ytr),
             "validation_accuracy":acc(model,Xva,yva),"test_accuracy":acc(model,Xte,yte),
             "leaves":ls,"mean_samples_per_leaf":mo,"singleton_leaf_fraction":sf}
        rows.append(row)
        if best is None or row["validation_accuracy"] > best[0]:
            best=(row["validation_accuracy"], {"max_depth":depth,"min_samples_leaf":leaf}, model)
    pd.DataFrame(rows).to_csv(RES/"tree_grid.csv",index=False)
    params=best[1]; tree=best[2]
    print("Selected tree:",params,"validation accuracy:",best[0])

    # Q1(b,c): bagging; all diagnostics computed on validation for disagreement/margin.
    bag_rows=[]; bags_by_B={}; rng=np.random.default_rng(SEED+1)
    for B in BAGS:
        start=time.time()
        trees,oob,coverage,distinct=fit_bag(Xtr,ytr,B,params,SEED+B)
        bags_by_B[B]=trees
        trpred,_=predict_vote(trees,Xtr,classes); vapred,vote= predict_vote(trees,Xva,classes)
        tepred,_=predict_vote(trees,Xte,classes)
        sorted_votes=np.sort(vote,axis=1)
        margin=float(np.mean(sorted_votes[:,-1] - sorted_votes[:,-2]))
        bag_rows.append({"B":B,"train_accuracy":float(np.mean(trpred==ytr)),
          "validation_accuracy":float(np.mean(vapred==yva)),"test_accuracy":float(np.mean(tepred==yte)),
          "oob_accuracy":oob,"oob_coverage":coverage,"mean_distinct_bootstrap":distinct,
          "theoretical_distinct":len(ytr)*(1-(1-1/len(ytr))**len(ytr)),
          "pairwise_disagreement":pair_disagreement(trees,Xva),"mean_vote_margin":margin,
          "seconds":time.time()-start})
        print(f"Bagging B={B}: val={bag_rows[-1]['validation_accuracy']:.3f}, OOB={oob:.3f}")
    bdf=pd.DataFrame(bag_rows); bdf.to_csv(RES/"bagging_grid.csv",index=False)
    # Highest validation accuracy, tie -> first B
    selected_B=int(bdf.loc[bdf.validation_accuracy.idxmax(),"B"])
    final_bag=bags_by_B[selected_B]
    print("Selected bagging B:",selected_B)

    # Q2(a) explicit 9-sample toy round, three classes.
    toy_X=np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8]],dtype=float)
    toy_y=np.array([0,0,0,1,1,1,2,2,2])
    toy_w=np.ones(9)/9
    toy=DecisionTreeClassifier(max_depth=1,random_state=SEED).fit(toy_X,toy_y,sample_weight=toy_w)
    toy_pred=toy.predict(toy_X); toy_err=float(toy_w[toy_pred!=toy_y].sum())
    toy_alpha=float(np.log((1-toy_err)/max(toy_err,1e-12))+np.log(2)) if toy_err < 2/3 else None
    toy_after=toy_w.copy()
    if toy_alpha is not None:
        toy_after *= np.exp(np.clip(toy_alpha*(toy_pred!=toy_y),-50,50)); toy_after/=toy_after.sum()
    (RES/"toy_samme.json").write_text(json.dumps({"X":toy_X.ravel().tolist(),"y":toy_y.tolist(),
        "initial_weights":toy_w.tolist(),"predictions":toy_pred.tolist(),"weighted_error":toy_err,
        "alpha":toy_alpha,"updated_weights":toy_after.tolist()},indent=2))

    # Q2(b): grid; cap accepted rounds at requested count.
    boost_rows=[]; boost_models={}
    for d,r in itertools.product(BOOST_DEPTHS,ROUNDS):
        st=time.time()
        model=SAMME(d,r,SEED+d*100+r).fit(Xtr,ytr)
        boost_models[(d,r)]=model
        boost_rows.append({"depth":d,"requested_rounds":r,"actual_rounds":len(model.models),
          "train_accuracy":acc(model,Xtr,ytr),"validation_accuracy":acc(model,Xva,yva),
          "test_accuracy":acc(model,Xte,yte),"seconds":time.time()-st})
        print(f"SAMME depth={d}, rounds={r}: val={boost_rows[-1]['validation_accuracy']:.3f}, actual={len(model.models)}")
    boostdf=pd.DataFrame(boost_rows); boostdf.to_csv(RES/"boosting_grid.csv",index=False)
    ix=boostdf.validation_accuracy.idxmax(); br=boostdf.loc[ix]
    selected_d=int(br.depth); selected_r=int(br.requested_rounds)
    final_boost=boost_models[(selected_d,selected_r)]
    pd.DataFrame(final_boost.history).to_csv(RES/"selected_boosting_history.csv",index=False)
    print("Selected SAMME:",selected_d,selected_r)

    # Final clean/masked comparison and confidence.
    models={"Single tree":tree,"Bagging":None,"SAMME":final_boost}
    pred_bag, prob_bag=predict_vote(final_bag,Xte,classes)
    _, prob_bag_mask=predict_vote(final_bag,Xm,classes)
    pred_tree=tree.predict(Xte); pred_tree_m=tree.predict(Xm)
    tree_conf=np.max(tree.predict_proba(Xte),axis=1); tree_conf_m=np.max(tree.predict_proba(Xm),axis=1)
    boost_prob=final_boost.proba_vote(Xte); boost_prob_m=final_boost.proba_vote(Xm)
    def summarize(name, clean_pred, mask_pred, clean_conf, mask_conf):
        return {"model":name,"clean_test_accuracy":float(np.mean(clean_pred==yte)),
          "masked_test_accuracy":float(np.mean(mask_pred==yte)),
          "clean_mean_confidence":float(np.mean(clean_conf)),
          "masked_mean_confidence":float(np.mean(mask_conf))}
    final_rows=[
      summarize("Single tree",pred_tree,pred_tree_m,tree_conf,tree_conf_m),
      summarize("Bagging",pred_bag,predict_vote(final_bag,Xm,classes)[0],np.max(prob_bag,axis=1),np.max(prob_bag_mask,axis=1)),
      summarize("SAMME",final_boost.predict(Xte),final_boost.predict(Xm),np.max(boost_prob,axis=1),np.max(boost_prob_m,axis=1))]
    pd.DataFrame(final_rows).to_csv(RES/"final_comparison.csv",index=False)

    # Q3: 30 within-identity bootstrap resamples; selected configurations frozen.
    # Prediction vectors: tree probabilities, bag vote fractions, SAMME normalised weighted scores.
    model_names=["Single tree","Bagging","SAMME"]
    vecs={n:[] for n in model_names}; accs={n:[] for n in model_names}
    depth_study=[]
    rng=np.random.default_rng(SEED+303)
    for rep in range(N_REPEATS):
        ids=[]
        for c in classes:
            class_ids=tr[ytr==c]
            ids.extend(rng.choice(class_ids,size=6,replace=True))
        ids=np.asarray(ids)
        rt=DecisionTreeClassifier(**params,random_state=SEED+rep).fit(X[ids],y[ids])
        rb,_,_,_=fit_bag(X[ids],y[ids],selected_B,params,SEED+1000+rep)
        rboost=SAMME(selected_d,selected_r,SEED+2000+rep).fit(X[ids],y[ids])
        tree_p=rt.predict_proba(Xte)
        # align sklearn class columns to global classes
        aligned=np.zeros((len(te),len(classes)))
        for j,c in enumerate(rt.classes_): aligned[:,np.where(classes==c)[0][0]]=tree_p[:,j]
        _, bag_p=predict_vote(rb,Xte,classes)
        boost_p=rboost.proba_vote(Xte)
        for n,p,pr in [("Single tree",aligned,rt.predict(Xte)),("Bagging",bag_p,predict_vote(rb,Xte,classes)[0]),("SAMME",boost_p,rboost.predict(Xte))]:
            vecs[n].append(p); accs[n].append(float(np.mean(pr==yte)))
        if rep in (0,): print("Bias/variance resampling started (30 refits)...")
    # Bias^2 and variance on fixed test set.
    onehot=np.stack([yte==c for c in classes],axis=1).astype(float)
    bv=[]
    for n in model_names:
        arr=np.stack(vecs[n]); mean=arr.mean(axis=0)
        bias2=float(np.mean(np.sum((mean-onehot)**2,axis=1)))
        variance=float(np.mean(np.sum((arr-mean[None,:,:])**2,axis=2)))
        bv.append({"model":n,"bias_squared":bias2,"variance":variance,
                   "refit_accuracy_mean":float(np.mean(accs[n])),"refit_accuracy_sd":float(np.std(accs[n],ddof=1))})
    pd.DataFrame(bv).to_csv(RES/"bias_variance.csv",index=False)

    # Depth study at shallow=2, selected depth (mapped to 16 if None), deep=16.
    depth_settings=[2,16,None]
    for dep in depth_settings:
        single_probs=[]; bag_probs=[]; single_acc=[]; bag_acc=[]
        for rep in range(N_REPEATS):
            ids=[]
            for c in classes: ids.extend(rng.choice(tr[ytr==c],size=6,replace=True))
            ids=np.asarray(ids)
            st=DecisionTreeClassifier(max_depth=dep,min_samples_leaf=params["min_samples_leaf"],random_state=SEED+5000+rep).fit(X[ids],y[ids])
            bt,_,_,_=fit_bag(X[ids],y[ids],100,{"max_depth":dep,"min_samples_leaf":params["min_samples_leaf"]},SEED+6000+rep)
            sp=np.zeros((len(te),len(classes))); p=st.predict_proba(Xte)
            for j,c in enumerate(st.classes_): sp[:,np.where(classes==c)[0][0]]=p[:,j]
            _,bp=predict_vote(bt,Xte,classes)
            single_probs.append(sp);bag_probs.append(bp)
            single_acc.append(acc(st,Xte,yte));bag_acc.append(float(np.mean(predict_vote(bt,Xte,classes)[0]==yte)))
        depth_study.append({"depth":str(dep),"single_bias_squared":float(np.mean(np.sum((np.mean(single_probs,axis=0)-onehot)**2,axis=1))),
          "single_variance":float(np.mean(np.sum((np.array(single_probs)-np.mean(single_probs,axis=0))**2,axis=2))),
          "single_accuracy_mean":float(np.mean(single_acc)),"single_accuracy_sd":float(np.std(single_acc,ddof=1)),
          "bag_bias_squared":float(np.mean(np.sum((np.mean(bag_probs,axis=0)-onehot)**2,axis=1))),
          "bag_variance":float(np.mean(np.sum((np.array(bag_probs)-np.mean(bag_probs,axis=0))**2,axis=2))),
          "bag_accuracy_mean":float(np.mean(bag_acc)),"bag_accuracy_sd":float(np.std(bag_acc,ddof=1))})
        print("Depth study complete:",dep)
    pd.DataFrame(depth_study).to_csv(RES/"depth_bias_variance.csv",index=False)

    # Plots
    plt.figure(); 
    tmp=pd.DataFrame(rows); tmp["depth_label"]=tmp.depth
    for leaf in LEAFS:
        sub=tmp[tmp.min_samples_leaf==leaf]; plt.plot(sub.depth,sub.validation_accuracy,marker="o",label=f"min leaf={leaf}")
    plt.xlabel("Maximum depth (None = unrestricted)");plt.ylabel("Validation accuracy");plt.legend();plt.tight_layout();plt.savefig(FIG/"tree_validation_grid.png",dpi=160);plt.close()
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for col,title in [("oob_accuracy","OOB accuracy"),("pairwise_disagreement","Pairwise disagreement"),("mean_vote_margin","Mean vote margin")]:
        axs[["oob_accuracy","pairwise_disagreement","mean_vote_margin"].index(col)].plot(bdf.B,bdf[col],marker="o")
        axs[["oob_accuracy","pairwise_disagreement","mean_vote_margin"].index(col)].set_xscale("log");axs[["oob_accuracy","pairwise_disagreement","mean_vote_margin"].index(col)].set_title(title);axs[["oob_accuracy","pairwise_disagreement","mean_vote_margin"].index(col)].set_xlabel("B")
    plt.tight_layout();plt.savefig(FIG/"bagging_diagnostics.png",dpi=160);plt.close()
    hist=pd.DataFrame(final_boost.history)
    fig,axs=plt.subplots(2,2,figsize=(9,6))
    for col in ["weighted_error","alpha","max_weight","top10_weight"]:
        ax=axs.flat[["weighted_error","alpha","max_weight","top10_weight"].index(col)];ax.plot(hist["round"],hist[col]);ax.set_title(col.replace("_"," ").title());ax.set_xlabel("Round")
    plt.tight_layout();plt.savefig(FIG/"samme_diagnostics.png",dpi=160);plt.close()
    bvdf=pd.DataFrame(bv)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    axs[0].bar(bvdf.model,bvdf.bias_squared);axs[0].set_title("Bias squared");axs[1].bar(bvdf.model,bvdf.variance);axs[1].set_title("Variance")
    for ax in axs: ax.tick_params(axis="x",rotation=20)
    plt.tight_layout();plt.savefig(FIG/"bias_variance.png",dpi=160);plt.close()

    summary={"student_name":"Kumar Gaurav","roll_number":ROLL,"seed":SEED,"split_sizes":{"train":len(tr),"validation":len(va),"test":len(te)},
      "selected_tree":params,"selected_bagging_B":selected_B,"selected_samme_depth":selected_d,"selected_samme_rounds":selected_r,
      "runtime_seconds":time.time()-t0,"note":"All metrics in results/ were computed by this run. Verify all output files before reporting."}
    (RES/"run_summary.json").write_text(json.dumps(summary,indent=2))
    print("\nDONE. Outputs saved in results/ and figures/. Runtime:",round(time.time()-t0,1),"seconds")
if __name__=="__main__": main()
