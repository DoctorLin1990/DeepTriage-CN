#!/usr/bin/env python3
"""
evaluation/metrics.py
=====================
Statistical evaluation functions — Section 3.8 of the paper.
"""
from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, precision_recall_curve,
    roc_auc_score, roc_curve, auc,
)
warnings.filterwarnings("ignore")

__all__ = [
    "bootstrap_auroc","delong_test","youden_metrics",
    "calibration_metrics","hosmer_lemeshow",
    "nri","alert_burden","interaction_contrast","full_evaluation_report",
]

# ─── helpers ────────────────────────────────────────────────────────────────

def _auc_boot(y,p,n_boot,rng):
    n=len(y); out=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n); yb,pb=y[idx],p[idx]
        if yb.sum()==0 or yb.sum()==n: continue
        out.append(roc_auc_score(yb,pb))
    return np.array(out)

# ─── public API ─────────────────────────────────────────────────────────────

def bootstrap_auroc(y,prob,n_bootstrap=1000,seed=42):
    rng   = np.random.default_rng(seed)
    auroc = roc_auc_score(y,prob)
    boot  = _auc_boot(y,prob,n_bootstrap,rng)
    prec,rec,_ = precision_recall_curve(y,prob)
    auprc = auc(rec,prec)
    n=len(y); ab=[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n); yb,pb=y[idx],prob[idx]
        if yb.sum()==0 or yb.sum()==n: continue
        pr,re,_=precision_recall_curve(yb,pb); ab.append(auc(re,pr))
    ab=np.array(ab)
    return dict(auroc=auroc, ci_lo=float(np.percentile(boot,2.5)),
                ci_hi=float(np.percentile(boot,97.5)), auprc=auprc,
                ci_auprc_lo=float(np.percentile(ab,2.5)) if len(ab) else np.nan,
                ci_auprc_hi=float(np.percentile(ab,97.5)) if len(ab) else np.nan)

def delong_test(y,prob_a,prob_b,n_bootstrap=1000,seed=42):
    rng=np.random.default_rng(seed)
    delta=roc_auc_score(y,prob_a)-roc_auc_score(y,prob_b)
    # Identical probability vectors → p=1.0 by definition
    if np.allclose(prob_a, prob_b):
        return dict(delta_auroc=0.0, ci_lo=0.0, ci_hi=0.0, p_value=1.0)
    n=len(y); deltas=[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n); yb,pa,pb=y[idx],prob_a[idx],prob_b[idx]
        if yb.sum()==0 or yb.sum()==n: continue
        deltas.append(roc_auc_score(yb,pa)-roc_auc_score(yb,pb))
    deltas=np.array(deltas)
    if len(deltas) == 0:
        return dict(delta_auroc=delta, ci_lo=np.nan, ci_hi=np.nan, p_value=np.nan)
    p=2*min((deltas>0).mean(),(deltas<0).mean()); p=max(p,1/n_bootstrap)
    return dict(delta_auroc=delta,
                ci_lo=float(np.percentile(deltas,2.5)),
                ci_hi=float(np.percentile(deltas,97.5)), p_value=float(p))

def youden_metrics(y,prob,threshold=None,n_bootstrap=1000,seed=42):
    rng=np.random.default_rng(seed)
    fpr_,tpr_,thr_=roc_curve(y,prob)
    if threshold is None: threshold=float(thr_[np.argmax(tpr_-fpr_)])
    def _m(yy,pp,t):
        pred=(pp>=t).astype(int)
        tp=int(((pred==1)&(yy==1)).sum()); tn=int(((pred==0)&(yy==0)).sum())
        fp=int(((pred==1)&(yy==0)).sum()); fn=int(((pred==0)&(yy==1)).sum())
        return dict(sens=tp/max(tp+fn,1), spec=tn/max(tn+fp,1),
                    acc=(tp+tn)/len(yy), ppv=tp/max(tp+fp,1),
                    alert_rate=pred.mean(), tp=tp,tn=tn,fp=fp,fn=fn)
    base=_m(y,prob,threshold); n=len(y)
    bs,bsp,ba,bp=[],[],[],[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n); m=_m(y[idx],prob[idx],threshold)
        bs.append(m["sens"]); bsp.append(m["spec"]); ba.append(m["acc"]); bp.append(m["ppv"])
    ci=lambda a:(float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
    return dict(threshold=threshold,
                sensitivity=base["sens"],  sens_ci=ci(bs),
                specificity=base["spec"],  spec_ci=ci(bsp),
                accuracy=base["acc"],      acc_ci=ci(ba),
                ppv=base["ppv"],           ppv_ci=ci(bp),
                alert_rate=base["alert_rate"],
                tp=base["tp"],tn=base["tn"],fp=base["fp"],fn=base["fn"])

def calibration_metrics(y,prob,n_bins=10,n_bootstrap=1000,seed=42):
    rng=np.random.default_rng(seed)
    brier=brier_score_loss(y,prob)
    lp=logit(np.clip(prob,1e-6,1-1e-6)).reshape(-1,1)
    lr=LogisticRegression(C=1e9,solver="lbfgs",max_iter=500)
    lr.fit(lp,y)
    slope=float(lr.coef_[0,0]); intercept=float(lr.intercept_[0])
    n=len(y); bb,bs,bi=[],[],[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n); yb,pb=y[idx],prob[idx]
        bb.append(brier_score_loss(yb,pb))
        try:
            lr_=LogisticRegression(C=1e9,solver="lbfgs",max_iter=200)
            lr_.fit(logit(np.clip(pb,1e-6,1-1e-6)).reshape(-1,1),yb)
            bs.append(float(lr_.coef_[0,0])); bi.append(float(lr_.intercept_[0]))
        except: pass
    ci=lambda a:(float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
    return dict(brier=brier,brier_ci=ci(bb),
                cal_slope=slope,slope_ci=ci(bs),
                cal_intercept=intercept,intercept_ci=ci(bi))

def hosmer_lemeshow(y,prob,n_bins=10):
    order=np.argsort(prob); ys=y[order]; ps=prob[order]
    groups=np.array_split(np.arange(len(ys)),n_bins); hl=0.0
    for g in groups:
        op=ys[g].sum(); ep=ps[g].sum(); on=len(g)-op; en=len(g)-ep
        if ep>0: hl+=(op-ep)**2/ep
        if en>0: hl+=(on-en)**2/en
    df=n_bins-2; p=float(1-stats.chi2.cdf(hl,df))
    return dict(hl_statistic=float(hl),df=df,p_value=p)

def nri(y,prob_new,prob_ref,n_bootstrap=1000,seed=42):
    rng=np.random.default_rng(seed)
    def _nri(yy,pn,pr):
        ev=yy==1; ne=yy==0
        return ((pn>pr)&ev).sum()/max(ev.sum(),1) - ((pn<pr)&ev).sum()/max(ev.sum(),1) + \
               ((pn<pr)&ne).sum()/max(ne.sum(),1) - ((pn>pr)&ne).sum()/max(ne.sum(),1)
    pt=_nri(y,prob_new,prob_ref); n=len(y)
    boot=[]; 
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n); boot.append(_nri(y[idx],prob_new[idx],prob_ref[idx]))
    boot=np.array(boot); se=float(boot.std(ddof=1)); z=pt/(se+1e-9)
    return dict(nri=float(pt),ci_lo=float(np.percentile(boot,2.5)),
                ci_hi=float(np.percentile(boot,97.5)),se=se,
                z=float(z),p_value=float(2*(1-stats.norm.cdf(abs(z)))))

def alert_burden(y,prob,thresholds=(0.15,0.28,0.40,0.50)):
    rows=[]
    for t in thresholds:
        pred=(prob>=t).astype(int)
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum())
        tn=int(((pred==0)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
        rows.append(dict(threshold=t,alert_rate=pred.mean(),
                         ppv=tp/max(tp+fp,1),sensitivity=tp/max(tp+fn,1),
                         specificity=tn/max(tn+fp,1),fn_rate=fn/max(tp+fn,1),
                         tp=tp,fp=fp,tn=tn,fn=fn))
    return pd.DataFrame(rows)

def interaction_contrast(y,prob_a,prob_b,group_mask,n_bootstrap=1000,seed=42):
    rng=np.random.default_rng(seed)
    def _d(yy,pa,pb,m):
        if m.sum()<5 or (~m).sum()<5: return np.nan
        return (roc_auc_score(yy[m],pa[m])-roc_auc_score(yy[m],pb[m])) - \
               (roc_auc_score(yy[~m],pa[~m])-roc_auc_score(yy[~m],pb[~m]))
    pt=_d(y,prob_a,prob_b,group_mask); n=len(y); boot=[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,n)
        v=_d(y[idx],prob_a[idx],prob_b[idx],group_mask[idx])
        if not np.isnan(v): boot.append(v)
    boot=np.array(boot); p=2*min((boot>0).mean(),(boot<0).mean()); p=max(p,1/n_bootstrap)
    return dict(delta_diff=float(pt),ci_lo=float(np.percentile(boot,2.5)),
                ci_hi=float(np.percentile(boot,97.5)),p_value=float(p),
                n_group1=int(group_mask.sum()),n_group0=int((~group_mask).sum()))

def full_evaluation_report(y,probs_dict,reference_models=("NEWS2","ESI"),
                           primary_model="DeepTriage-CN",youden_threshold=0.28,
                           geri_mask=None,n_bootstrap=1000,seed=42):
    report={}
    for name,prob in probs_dict.items():
        report[name]={
            **bootstrap_auroc(y,prob,n_bootstrap,seed),
            **youden_metrics(y,prob,youden_threshold,n_bootstrap,seed),
            **calibration_metrics(y,prob,n_bootstrap=n_bootstrap,seed=seed),
            **hosmer_lemeshow(y,prob),
            "delong_vs_primary": delong_test(y,probs_dict[primary_model],prob,n_bootstrap,seed),
        }
    for ref in reference_models:
        if ref in probs_dict and primary_model in probs_dict:
            report[primary_model][f"nri_vs_{ref}"]=nri(y,probs_dict[primary_model],probs_dict[ref],n_bootstrap,seed)
    if primary_model in probs_dict:
        report[primary_model]["alert_burden"]=alert_burden(y,probs_dict[primary_model]).to_dict(orient="records")
    if geri_mask is not None:
        gr={}; y_g=y[geri_mask]
        for name,prob in probs_dict.items():
            if geri_mask.sum()<10: continue
            p_g=prob[geri_mask]
            gr[name]={**bootstrap_auroc(y_g,p_g,n_bootstrap,seed),
                      **calibration_metrics(y_g,p_g,n_bootstrap=n_bootstrap,seed=seed),
                      **hosmer_lemeshow(y_g,p_g)}
        report["_geriatric_subgroup"]=gr
        ics={}
        for ref in list(reference_models)+["TabNet"]:
            if ref in probs_dict:
                ics[f"{primary_model}_vs_{ref}_age"]=interaction_contrast(
                    y,probs_dict[primary_model],probs_dict[ref],geri_mask,n_bootstrap,seed)
        report["_interactions"]=ics
    return report
