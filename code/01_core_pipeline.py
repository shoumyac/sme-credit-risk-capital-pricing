from common import *
import json
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
V={}
n=len(y); V["n"]=n; V["n_def"]=int(y.sum()); V["prev"]=100*y.mean()
V["auc_pooled"]=roc_auc_score(y,raw); V["raw_mean_pd"]=100*raw.mean(); V["raw_brier"]=brier_score_loss(y,raw)
V["slope_pooled"],V["int_pooled"]=slope_int(raw,y)
# nested PDs
PD_N=np.zeros(n); PD_NI=np.zeros(n)
for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(raw.reshape(-1,1),y):
    PD_N[te]=platt_pred(platt_fit(raw[tr],y[tr]),raw[te])
    PD_NI[te]=np.clip(IsotonicRegression(out_of_bounds="clip").fit(raw[tr],y[tr]).predict(raw[te]),0.0003,0.9999)
def prof(rt,P,mask): return (((rt-RF_-SF-CO)*(1-P)-P*LGD)*EAD*mask).sum()/1e6
def core(P,tag,lgd=LGD):
    K=basel_K(P,lgd); r=rate(P,lgd,K=K); v=r<=CAP
    d=dict(tag=tag,mean_pd=100*P.mean(),brier=brier_score_loss(y,P),logloss=log_loss(y,P),auc=roc_auc_score(y,P),distinct=len(np.unique(P.round(8))),
      at_floor=int((P<=0.0003+1e-12).sum()),mean_K=100*K.mean(),mean_rwa_m=(K*12.5*EAD).mean()/1e6,mean_t1_k=(0.06*K*12.5*EAD).mean()/1e3,total_t1=(0.06*K*12.5*EAD).sum()/1e6,
      median_rate=100*np.median(r),viable=100*v.mean(),credit_m=EAD[v].sum()/1e6,pool_default=100*y[v].mean(),pool_el=(P[v]*lgd*EAD[v]).sum()/1e6)
    return d,K,r,v
dP,K_P,r_P,v_P=core(PD_P:=PD_platt,"Platt"); dI,K_I,r_I,v_I=core(PD_iso,"Isotonic"); dN,K_N,r_N,v_N=core(PD_N,"Platt nested"); dNI,_,_,v_NI=core(PD_NI,"Isotonic nested")
RC=pd.DataFrame([dI,dP,dN,dNI]); RC["raw_row"]=0
raw_row=dict(tag="Raw balanced score",mean_pd=100*raw.mean(),brier=brier_score_loss(y,raw),logloss=log_loss(y,raw),auc=roc_auc_score(y,raw),distinct=len(np.unique(raw.round(8))))
RC=pd.concat([pd.DataFrame([raw_row]),RC],ignore_index=True); RC.to_csv(OUT+"b_recalibrators.csv",index=False)
V.update({f"P_{k}":v for k,v in dP.items() if k!="tag"}); V.update({f"I_{k}":v for k,v in dI.items() if k!="tag"}); V.update({f"N_{k}":v for k,v in dN.items() if k!="tag"}); V.update({f"NI_{k}":v for k,v in dNI.items() if k!="tag"})
V["brier_gain_pct"]=100*(1-dP["brier"]/V["raw_brier"]); V["t1_iso_vs_platt_pct"]=100*(dI["total_t1"]/dP["total_t1"]-1)  # iso lower
V["iso_floor_pct"]=100*dI["at_floor"]/n; V["iso_floor_defaults"]=int(y[PD_iso<=0.0003+1e-12].sum())
# regime comparison (Platt PD, nested screen)
rows=[]
for nm,rt,mask,P in (("Flat 18%",np.full(n,0.18),np.ones(n,bool),PD_platt),("RAROC price, in-sample screen",np.minimum(r_P,CAP),v_P,PD_platt),
                     ("RAROC price, nested screen",np.minimum(r_N,CAP),v_N,PD_N),("Flat 18% price, nested screen",np.full(n,0.18),v_N,PD_N)):
    rows.append(dict(regime=nm,approval=100*mask.mean(),default=100*y[mask].mean(),volume_m=EAD[mask].sum()/1e6,el_m=(P[mask]*LGD*EAD[mask]).sum()/1e6,profit_m=prof(rt,P,mask)))
RG=pd.DataFrame(rows); RG.to_csv(OUT+"b_regime.csv",index=False)
V["flat_loss"]=-RG.profit_m[0]; V["raroc_profit"]=RG.profit_m[1]; V["raroc_profit_nested"]=RG.profit_m[2]
# sensitivity grid
rows=[]
for lgd in (0.30,0.45,0.60,0.75):
    Kl=basel_K(PD_platt,lgd)
    for h in np.round(np.arange(0.05,0.2501,0.01),2):
        rt=RF_+SF+CO+(PD_platt*lgd+h*Kl)/(1-PD_platt); rows.append(dict(LGD=lgd,hurdle=h,viable=100*(rt<=CAP).mean(),median_rate=100*np.median(rt)))
SG=pd.DataFrame(rows); SG.to_csv(OUT+"b_sens_grid.csv",index=False)
# cap and funding-rate sensitivity
rows=[]
for cap in (0.18,0.20,0.22,0.24,0.26,0.30):
    rows.append(dict(kind="cap",value=cap,viable=100*(r_P<=cap).mean()))
for rf in (0.07,0.09,0.11):
    rt=rf+SF+CO+(PD_platt*LGD+H*K_P)/(1-PD_platt); rows.append(dict(kind="rf",value=rf,viable=100*(rt<=CAP).mean()))
pd.DataFrame(rows).to_csv(OUT+"b_cap_rf.csv",index=False)
# aggregates by sector, division
rows=[]
for a in ("sector","division","location_type"):
    for g in sorted(df[a].unique()):
        m=(df[a]==g).values
        rows.append(dict(attribute=a,group=g,n=int(m.sum()),pd=100*PD_platt[m].mean(),obs=100*y[m].mean(),K=100*K_P[m].mean(),t1_m=(0.06*K_P[m]*12.5*EAD[m]).sum()/1e6,
                         median_rate=100*np.median(r_P[m]),p75_rate=100*np.percentile(r_P[m],75),viable=100*v_P[m].mean()))
AG=pd.DataFrame(rows); AG.to_csv(OUT+"b_groups.csv",index=False)
# treatment
rows=[]
for tr,lab in (("corporate_sme","Corporate SME formula"),("retail_other","Other retail formula")):
    rec={"treatment":lab}
    for tag,P in (("iso",PD_iso),("platt",PD_platt)):
        K=basel_K(P,treatment=tr); r=rate(P,K=K); rec[f"K_{tag}"]=100*K.mean(); rec[f"t1_{tag}"]=(0.06*K*12.5*EAD).sum()/1e6; rec[f"viable_{tag}"]=100*(r<=CAP).mean(); rec[f"rate_{tag}"]=100*np.median(r)
    rows.append(rec)
for rw,lab in ((0.75,"Standardised, 75% RW"),(1.0,"Standardised, 100% RW")):
    rec={"treatment":lab}
    for tag,P in (("iso",PD_iso),("platt",PD_platt)):
        K=np.full(n,0.08*rw); r=rate(P,K=K); rec[f"K_{tag}"]=100*K.mean(); rec[f"t1_{tag}"]=(0.06*rw*EAD).sum()/1e6; rec[f"viable_{tag}"]=100*(r<=CAP).mean(); rec[f"rate_{tag}"]=100*np.median(r)
    rows.append(rec)
TR=pd.DataFrame(rows); TR.to_csv(OUT+"b_treatment.csv",index=False)
V["retail_drop_pct"]=100*(TR.K_platt[1]/TR.K_platt[0]-1); V["retail_drop_pct_iso"]=100*(TR.K_iso[1]/TR.K_iso[0]-1)
V["treat_ratio"]=TR.t1_platt.max()/TR.t1_platt.min()
# default definition
def pshift(p,a,b):
    o=p/(1-p)*(b/(1-b))*((1-a)/a); return np.clip(o/(1+o),0.0003,0.9999)
rows=[]; pi0=PD_platt.mean()
for phi in (1.0,0.7,0.5,0.3,0.2):
    P=pshift(PD_platt,pi0,phi*pi0); K=basel_K(P); r=rate(P,K=K); rows.append(dict(phi=phi,rate=100*phi*pi0,K=100*K.mean(),t1=(0.06*K*12.5*EAD).sum()/1e6,median_rate=100*np.median(r),viable=100*(r<=CAP).mean()))
DD=pd.DataFrame(rows); DD.to_csv(OUT+"b_defdef.csv",index=False); V["dd_span"]=DD.viable.max()-DD.viable.min()
# guarantee
rows=[]; base=None
for cov in (0,0.2,0.4,0.5,0.6,0.8):
    lg=LGD*(1-cov); K=basel_K(PD_platt,lg); r=rate(PD_platt,lg,K=K); v=r<=CAP
    if base is None: base=EAD[v].sum()/1e6
    rows.append(dict(coverage=100*cov,K=100*K.mean(),median_rate=100*np.median(r),viable=100*v.mean(),credit=EAD[v].sum()/1e6,incr=EAD[v].sum()/1e6-base,payout=(PD_platt*LGD*cov*EAD).sum()/1e6))
r_el=RF_+SF+CO+PD_platt*LGD; vel=r_el<=CAP
rows.append(dict(coverage=np.nan,K=0.0,median_rate=100*np.median(r_el),viable=100*vel.mean(),credit=EAD[vel].sum()/1e6,incr=EAD[vel].sum()/1e6-base,payout=0.0))
GU=pd.DataFrame(rows); GU.to_csv(OUT+"b_guarantee.csv",index=False); V["ul_incr"]=GU.incr.iloc[-1]; V["ul_viable"]=GU.viable.iloc[-1]
V["leverage_lo"]=(GU.incr[1:6]/GU.payout[1:6]).min(); V["leverage_hi"]=(GU.incr[1:6]/GU.payout[1:6]).max()
json.dump({k:float(v) for k,v in V.items()},open(OUT+"V.json","w"),indent=1)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
for t in (RC,RG,TR,DD,GU,AG): print(t.round(3).to_string(index=False)); print()
print({k:round(float(v),3) for k,v in V.items()})
