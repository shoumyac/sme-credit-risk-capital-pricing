import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import external_helpers as H, common as C
from importlib import import_module
load=import_module('06_external_replication').load
rng=np.random.default_rng(5)
X,y,G,tr=load("Taiwan (card default)"); o=H.get_oof(X,y,"balanced"); ead=np.ones(len(y)); rs=[]
for n in (500,1000,2000,5000,30000):
    for rep in range(10 if n<30000 else 1):
        j=rng.choice(len(y),n,replace=False) if n<30000 else np.arange(len(y)); oo=o[j]; yy=y[j]
        pl=LogisticRegression(C=1e6,max_iter=5000).fit(C.lo(oo).reshape(-1,1),yy); Pp=np.clip(pl.predict_proba(C.lo(oo).reshape(-1,1))[:,1],0.0003,0.9999)
        Pi=np.clip(IsotonicRegression(out_of_bounds="clip").fit(oo,yy).predict(oo),0.0003,0.9999)
        a=H.metr(Pi,yy,ead[j],tr); b=H.metr(Pp,yy,ead[j],tr)
        rs.append(dict(n=n,iso_distinct=a["distinct"],iso_at_floor_pct=100*a["at_floor"]/n,tier1_ratio_iso_over_platt=a["tier1_per_unit"]/b["tier1_per_unit"],viable_iso=a["share_viable"],viable_platt=b["share_viable"]))
R=pd.DataFrame(rs).groupby("n").mean().reset_index(); R.to_csv(C.OUT+"a5_isotonic_vs_n.csv",index=False); print(R.round(3))
