from common import *
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
rng=np.random.default_rng(11)
def tier1(P,lgd=LGD): K=basel_K(P,lgd); return (0.06*K*12.5*EAD).sum()/1e6,K.mean(),(rate(P,lgd,K=K)<=CAP).mean()
def ece5(p,yy):
    q=np.quantile(p,np.linspace(0,1,6)); idx=np.clip(np.searchsorted(q,p,side="right")-1,0,4)
    return sum((idx==b).mean()*abs(p[idx==b].mean()-yy[idx==b].mean()) for b in range(5) if (idx==b).any())

# ---------- A3 class weighting / prior shift ----------
print("=== A3 class weighting"); rows=[]
pi=y.mean()
for seed in range(5):
    for cw in ("balanced",None):
        o=oof(df,y,cw=cw,seed=seed)
        # raw used as PD directly (the 'naive' route), then Platt, then analytic prior correction (balanced only)
        variants={"raw as PD":np.clip(o,0.0003,0.9999),"Platt":platt_pred(platt_fit(o,y),o)}
        if cw=="balanced":
            odds=o/(1-o)*pi/(1-pi); variants["analytic prior correction"]=np.clip(odds/(1+odds),0.0003,0.9999)
        for v,P in variants.items():
            t1,Km,sv=tier1(P)
            rows.append(dict(seed=seed,weighting=cw or "none",variant=v,mean_pd=P.mean(),auc=roc_auc_score(y,P),brier=brier_score_loss(y,P),ece=ece5(P,y),tier1_m=t1,mean_K=100*Km,share_viable=100*sv))
A3=pd.DataFrame(rows); A3.to_csv(OUT+"a3_class_weighting_runs.csv",index=False)
A3s=A3.groupby(["weighting","variant"]).mean().drop(columns="seed").reset_index(); A3s.to_csv(OUT+"a3_class_weighting.csv",index=False)
print(A3s.round(4).to_string(index=False))
# prevalence shift grid on the Platt PD (prior-shift correction)
rows=[]
for pn in (0.05,0.10,0.15,0.2525,0.35):
    odds=PD_platt/(1-PD_platt)*(pn/(1-pn))/(pi/(1-pi)); P=np.clip(odds/(1+odds),0.0003,0.9999)
    t1,Km,sv=tier1(P); rows.append(dict(prevalence=pn,mean_pd=P.mean(),tier1_m=t1,mean_K=100*Km,share_viable=100*sv))
PS=pd.DataFrame(rows); PS.to_csv(OUT+"a3_prevalence_shift.csv",index=False); print(PS.round(3).to_string(index=False))

# ---------- A4 truncation experiment (reject-inference) ----------
print("=== A4 truncation")
late={"Never":0,"Rarely":1,"Sometimes":2,"Often":3}
colm={"Low":0,"Moderate":1,"High":2}
hs=(df.monthly_profit_bdt/df.loan_amount_bdt).rank(pct=True).values+df.collateral_level.map(colm).values/2-df.late_payment_history.map(late).fillna(1).values/3
Pfull=PD_platt
rows=[]
def fit_eval(tr,te,w=None):
    p=make_pipe("balanced").fit(df.iloc[tr],y[tr],**({"clf__sample_weight":w} if w is not None else {}))
    s_tr=np.zeros(len(tr))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=1).split(tr,y[tr]):
        pp=make_pipe("balanced").fit(df.iloc[tr[a]],y[tr[a]],**({"clf__sample_weight":w[a]} if w is not None else {})); s_tr[b]=pp.predict_proba(df.iloc[tr[b]])[:,1]
    pl=platt_fit(s_tr,y[tr]); return platt_pred(pl,p.predict_proba(df.iloc[te])[:,1])
for pol in ("scorecard","random"):
    for q in (0.1,0.2,0.3,0.4,0.5):
        if pol=="scorecard": thr=np.quantile(hs,q); acc=np.where(hs>=thr)[0]; rej=np.where(hs<thr)[0]
        else:
            perm=rng.permutation(len(y)); rej=perm[:int(q*len(y))]; acc=perm[int(q*len(y)):]
        for corr in ("none","IPW"):
            w=None
            if corr=="IPW":
                Xp=pd.get_dummies(df[CAT]).values.astype(float); Xn=(df[NUM]-df[NUM].median()); Xn=((Xn-Xn.mean())/Xn.std()).values
                Z=np.c_[Xn,Xp]; ya=np.zeros(len(y)); ya[acc]=1
                pr=LogisticRegression(C=0.5,max_iter=5000).fit(Z,ya).predict_proba(Z)[acc,1]; w=1/np.clip(pr,0.05,1); w=w/w.mean()
            P=fit_eval(acc,rej,w)
            t_est=(0.06*basel_K(P,LGD,S[rej],M[rej])*12.5*EAD[rej]).sum()/1e6; t_ref=(0.06*basel_K(Pfull[rej],LGD,S[rej],M[rej])*12.5*EAD[rej]).sum()/1e6
            rows.append(dict(policy=pol,reject_share=q,correction=corr,n_rej=len(rej),obs_default_rej=y[rej].mean(),mean_pd_rej=P.mean(),
                auc_rej=roc_auc_score(y[rej],P),brier_rej=brier_score_loss(y[rej],P),ece_rej=ece5(P,y[rej]),cil_rej=P.mean()-y[rej].mean(),
                tier1_est=t_est,tier1_ref=t_ref,tier1_err_pct=100*(t_est/t_ref-1)))
A4=pd.DataFrame(rows); A4.to_csv(OUT+"a4_truncation.csv",index=False); print(A4.round(3).to_string(index=False))

# ---------- A5 thresholded fairness ----------
print("=== A5 thresholded fairness")
GM=group_masks()
r=rate(PD_platt); app=r<=CAP
def fair_rows(app,tag):
    out=[]
    for a,g,m in GM:
        good=m&(y==0); bad=m&(y==1)
        out.append(dict(variant=tag,attribute=a,group=g,n=int(m.sum()),approval=app[m].mean(),good_approval=app[good].mean(),bad_approval=app[bad].mean() if bad.any() else np.nan,
                        default_in_approved=y[m&app].mean() if (m&app).any() else np.nan))
    return pd.DataFrame(out)
F=fair_rows(app,"RAROC, global Platt"); 
def gaps(app):
    o={}
    for a in GROUPS:
        gs=[(g,m) for aa,g,m in GM if aa==a]
        ap=[app[m].mean() for g,m in gs]; ga=[app[m&(y==0)].mean() for g,m in gs]
        o[a]=(max(ap)-min(ap),max(ga)-min(ga),min(ap)/max(ap))
    return o
base=gaps(app); B=1000; bs={a:[] for a in GROUPS}
for _ in range(B):
    j=rng.integers(0,len(y),len(y))
    pl=platt_fit(raw[j],y[j]); P=platt_pred(pl,raw); ap=rate(P)<=CAP
    # evaluate on resample
    o={}
    for a in GROUPS:
        gs=[(g,(df[a].values[j]==g)) for aa,g,m in GM if aa==a]
        apx=[ap[j][m].mean() for g,m in gs]; gax=[ap[j][m&(y[j]==0)].mean() for g,m in gs]
        bs[a].append((max(apx)-min(apx),max(gax)-min(gax),min(apx)/max(apx)))
rows=[]
for a in GROUPS:
    b=np.array(bs[a]); lo_,hi_=np.percentile(b,[2.5,97.5],axis=0)
    rows.append(dict(attribute=a,approval_gap=base[a][0],approval_gap_lo=lo_[0],approval_gap_hi=hi_[0],good_gap=base[a][1],good_gap_lo=lo_[1],good_gap_hi=hi_[1],
        impact_ratio=base[a][2],ratio_lo=lo_[2],ratio_hi=hi_[2]))
FG=pd.DataFrame(rows); FG.to_csv(OUT+"a5_fairness_gaps.csv",index=False); F.to_csv(OUT+"a5_fairness_groups.csv",index=False)
print(F.round(3).to_string(index=False)); print(FG.round(3).to_string(index=False))

# ---------- A6 take-up / adverse selection ----------
print("=== A6 take-up")
def profit_demand(rt,mask,k,gamma,P=PD_platt,r0=0.18):
    take=1/(1+np.exp(k*(rt-r0)*100)); od=P/(1-P)*np.exp(gamma*(rt-r0)*100); Pa=od/(1+od)   # gamma per percentage point
    prof=(((rt-RF_-SF-CO)*(1-Pa)-Pa*LGD)*EAD*take*mask).sum()/1e6; vol=(EAD*take*mask).sum()/1e6
    return prof,vol,(Pa*take*mask*EAD).sum()/ (take*mask*EAD).sum()
rows=[]
rr=np.minimum(rate(PD_platt),CAP); v=rate(PD_platt)<=CAP
for k in (0.0,0.15,0.3):
    for g in (0.0,0.03,0.06):
        pf,vf,df_=profit_demand(np.full(len(y),0.18),np.ones(len(y),bool),k,g)
        pr,vr,dr=profit_demand(rr,v,k,g)
        rows.append(dict(takeup_slope=k,adverse_gamma=g,flat_profit=pf,flat_volume=vf,flat_pd=df_,raroc_profit=pr,raroc_volume=vr,raroc_pd=dr))
A6=pd.DataFrame(rows); A6.to_csv(OUT+"a6_takeup.csv",index=False); print(A6.round(2).to_string(index=False))

# ---------- A7 downturn ----------
print("=== A7 downturn")
rows=[]
for lam in (1.0,1.5,2.0,3.0):
    for lg in (0.45,0.55):
        odds=PD_platt/(1-PD_platt)*lam; P=np.clip(odds/(1+odds),0.0003,0.9999); K=basel_K(P,lg); rt=rate(P,lg,K=K)
        rows.append(dict(odds_multiplier=lam,lgd=lg,mean_pd=P.mean(),tier1_m=(0.06*K*12.5*EAD).sum()/1e6,share_viable=100*(rt<=CAP).mean(),median_rate=100*np.median(rt)))
A7=pd.DataFrame(rows); A7.to_csv(OUT+"a7_downturn.csv",index=False); print(A7.round(3).to_string(index=False))

# ---------- A9 collateral-linked LGD ----------
print("=== A9 LGD by collateral")
sets={"flat 0.45":{"Low":.45,"Moderate":.45,"High":.45},"collateral-linked (0.60/0.45/0.30)":{"Low":.60,"Moderate":.45,"High":.30},"steeper (0.75/0.45/0.20)":{"Low":.75,"Moderate":.45,"High":.20}}
rows=[]
for nm,mp in sets.items():
    lg=df.collateral_level.map(mp).values.astype(float); K=basel_K(PD_platt,lg); rt=rate(PD_platt,lg,K=K)
    for c in ("Low","Moderate","High"):
        m=(df.collateral_level==c).values
        rows.append(dict(lgd_scheme=nm,collateral=c,n=int(m.sum()),lgd=mp[c],default_rate=y[m].mean(),mean_K=100*K[m].mean(),share_viable=100*(rt[m]<=CAP).mean(),median_rate=100*np.median(rt[m])))
    rows.append(dict(lgd_scheme=nm,collateral="ALL",n=len(y),lgd=lg.mean(),default_rate=y.mean(),mean_K=100*K.mean(),share_viable=100*(rt<=CAP).mean(),median_rate=100*np.median(rt),tier1_m=(0.06*K*12.5*EAD).sum()/1e6))
A9=pd.DataFrame(rows); A9.to_csv(OUT+"a9_lgd_collateral.csv",index=False); print(A9.round(3).to_string(index=False))
