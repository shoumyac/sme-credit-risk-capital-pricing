from common import *
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(2026)
GM=group_masks()
B=1000
def ece5(p,yy):
    q=np.quantile(p,np.linspace(0,1,6)); idx=np.clip(np.searchsorted(q,p,side="right")-1,0,4)
    return sum((idx==b).mean()*abs(p[idx==b].mean()-yy[idx==b].mean()) for b in range(5) if (idx==b).any())
def stats_(p,yy):
    s,i=slope_int(p,yy); return s,p.mean()-yy.mean(),ece5(p,yy),(roc_auc_score(yy,p) if 0<yy.sum()<len(yy) else np.nan)
preds={"raw balanced LR":raw,"global Platt":PD_platt}
rows=[]
for nm,P in preds.items():
    for a,g,m in GM:
        idx=np.where(m)[0]; n=len(idx)
        obs=np.array(stats_(P[idx],y[idx]))
        bs=[]
        for _ in range(B):
            j=rng.choice(idx,n); bs.append(stats_(P[j],y[j]))
        bs=np.array(bs)
        lo_,hi_=np.nanpercentile(bs,[2.5,97.5],axis=0)
        # CIL z-test under H0 calibrated in group
        var=(P[idx]*(1-P[idx])).sum()/n**2; z=(y[idx].mean()-P[idx].mean())/np.sqrt(var)
        rows.append(dict(model=nm,attribute=a,group=g,n=n,obs=y[idx].mean(),pred=P[idx].mean(),
          cil=obs[1],cil_lo=lo_[1],cil_hi=hi_[1],slope=obs[0],slope_lo=lo_[0],slope_hi=hi_[0],
          ece=obs[2],ece_lo=lo_[2],ece_hi=hi_[2],auc=obs[3],auc_lo=lo_[3],auc_hi=hi_[3],cil_z=z,
          cil_p=2*(1-norm.cdf(abs(z)))))
T=pd.DataFrame(rows)
# Holm within model
for nm in preds:
    m=T.model==nm; p=T.loc[m,"cil_p"].values; o=np.argsort(p); k=len(p); adj=np.empty(k); run=0
    for r,i in enumerate(o): run=max(run,min(1,(k-r)*p[i])); adj[i]=run
    T.loc[m,"cil_p_holm"]=adj
T.to_csv(OUT+"a1_subgroup_bootstrap.csv",index=False)
# null calibration: does the dispersion of subgroup slope / CIL exceed pure sampling noise?
res=[]
for nm,P in preds.items():
    t=T[T.model==nm]
    obs_rng_s=t.slope.max()-t.slope.min(); obs_rng_c=t.cil.max()-t.cil.min(); obs_rng_a=t.auc.max()-t.auc.min()
    ns,nc,na=[],[],[]
    for _ in range(B):
        ys=(rng.random(len(P))<P).astype(int)
        # null: perfectly calibrated model in every group (and AUC same ranking)
        ss,cc,aa=[],[],[]
        for a,g,m in GM:
            s,c,e,au=stats_(P[m],ys[m]); ss.append(s); cc.append(c); aa.append(au)
        ns.append(max(ss)-min(ss)); nc.append(max(cc)-min(cc)); na.append(np.nanmax(aa)-np.nanmin(aa))
    ns,nc,na=map(np.array,(ns,nc,na))
    res.append(dict(model=nm,slope_range_obs=obs_rng_s,slope_range_null_median=np.median(ns),slope_range_null_p95=np.percentile(ns,95),p_slope=(ns>=obs_rng_s).mean(),
      cil_range_obs=obs_rng_c,cil_range_null_median=np.median(nc),cil_range_null_p95=np.percentile(nc,95),p_cil=(nc>=obs_rng_c).mean(),
      auc_range_obs=obs_rng_a,auc_range_null_median=np.median(na),auc_range_null_p95=np.percentile(na,95),p_auc=(na>=obs_rng_a).mean()))
R=pd.DataFrame(res); R.to_csv(OUT+"a1_null_dispersion.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(T[T.model=="raw balanced LR"][["attribute","group","n","cil","cil_lo","cil_hi","slope","slope_lo","slope_hi","cil_p_holm"]].round(3).to_string(index=False))
print(T[T.model=="global Platt"][["attribute","group","n","cil","cil_lo","cil_hi","slope","slope_lo","slope_hi","cil_p_holm"]].round(3).to_string(index=False))
print(R.round(4).T)
