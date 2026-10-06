import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
import common as C
def pipe(X,cw):
    num=[c for c in X.columns if X[c].dtype.kind in "if" and c not in ("age_band",)]; cat=[c for c in X.columns if c not in num]
    return Pipeline([("p",ColumnTransformer([("n",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
        ("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])),
        ("l",LogisticRegression(max_iter=5000,class_weight=cw,C=1.0))])
def get_oof(X,y,cw,seed=42):
    o=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        o[te]=pipe(X,cw).fit(X.iloc[tr],y[tr]).predict_proba(X.iloc[te])[:,1]
    return o
def metr(P,y,ead,tr='retail_other'):
    K=C.basel_K(P,C.LGD,np.full_like(P,5.0),np.full_like(P,2.5),tr); r=C.RF_+C.SF+C.CO+(P*C.LGD+C.H*K)/(1-P)
    return dict(mean_pd=P.mean(),brier=brier_score_loss(y,P),tier1_per_unit=(0.06*K*12.5*ead).sum()/ead.sum(),share_viable=100*(r<=C.CAP).mean(),
                distinct=len(np.unique(P.round(8))),at_floor=int((P<=0.0003+1e-12).sum()))
def ece5(p,yy):
    q=np.quantile(p,np.linspace(0,1,6)); idx=np.clip(np.searchsorted(q,p,side="right")-1,0,4)
    return sum((idx==b).mean()*abs(p[idx==b].mean()-yy[idx==b].mean()) for b in range(5) if (idx==b).any())
