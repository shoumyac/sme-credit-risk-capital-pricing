from common import *
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(99); B=1000
def headline(P,yy,idx):
    K=basel_K(P[idx],LGD,S[idx],M[idx]); r=rate(P[idx],K=K); v=r<=CAP
    ead=EAD[idx]; t1=(0.06*K*12.5*ead).sum()/len(idx)*1200/1e6
    flat=(((0.18-RF_-SF-CO)*(1-P[idx])-P[idx]*LGD)*ead).sum()/len(idx)*1200/1e6
    return dict(auc=roc_auc_score(yy[idx],raw[idx]),mean_K=100*K.mean(),tier1_m=t1,share_viable=100*v.mean(),default_in_pool=100*yy[idx][v].mean(),flat_profit_m=flat,median_rate=100*np.median(r))
rows=[]
for b in range(B):
    j=rng.integers(0,len(y),len(y)); pl=platt_fit(raw[j],y[j]); P=platt_pred(pl,raw)
    rows.append(headline(P,y,j))
bs=pd.DataFrame(rows)
base=headline(PD_platt,y,np.arange(len(y)))
out=pd.DataFrame({"estimate":pd.Series(base),"lo":bs.quantile(.025),"hi":bs.quantile(.975)}); out.to_csv(OUT+"a8_headline_ci.csv"); print(out.round(3))
# subgroup capital/price bound if hidden miscalibration equals the CI endpoint of group CIL
T=pd.read_csv(OUT+"a1_subgroup_bootstrap.csv"); T=T[T.model=="global Platt"]
rows=[]
for _,t in T.iterrows():
    m=(df[t.attribute]==t.group).values
    K0=basel_K(PD_platt[m],LGD,S[m],M[m]); r0=rate(PD_platt[m],K=K0)
    for lab,sh in (("CIL upper",t.cil_hi),("CIL lower",t.cil_lo)):
        odds_shift=lambda P,sh=sh:np.clip(P-sh,0.0003,0.9999)   # true PD = predicted minus CIL (pred - obs = CIL)
        Pt=odds_shift(PD_platt[m]); K1=basel_K(Pt,LGD,S[m],M[m]); r1=rate(Pt,K=K1)
        rows.append(dict(attribute=t.attribute,group=t.group,n=t.n,bound=lab,cil=sh,capital_misstatement_pct=100*(K0.mean()/K1.mean()-1),rate_diff_pp=100*(np.median(r0)-np.median(r1))))
Bd=pd.DataFrame(rows); Bd.to_csv(OUT+"a2_capital_price_bound.csv",index=False)
print(Bd.round(2).to_string(index=False))
