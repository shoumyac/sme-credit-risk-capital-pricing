from common import *
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
import itertools
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
rng=np.random.default_rng(7)
ATTR=["sector","division","location_type","owner_education"]
lr_=lo(raw)
def fit_pred(tr,te,mode,attr):
    x=lr_.reshape(-1,1)
    if mode=="platt":
        m=LogisticRegression(C=1e6,max_iter=5000).fit(x[tr],y[tr]); return m.predict_proba(x[te])[:,1]
    if mode=="iso":
        return IsotonicRegression(out_of_bounds="clip").fit(raw[tr],y[tr]).predict(raw[te])
    D=pd.get_dummies(df[attr]).values.astype(float)
    if mode.startswith("offset"):
        C={"offset":1e6,"offset_shrunk":1.0}[mode]
        X=np.c_[x,D]; m=LogisticRegression(C=C,max_iter=5000).fit(X[tr],y[tr]); return m.predict_proba(X[te])[:,1]
    if mode=="group_platt":
        out=np.zeros(len(te)); te_=np.asarray(te)
        for g in range(D.shape[1]):
            gt=tr[D[tr,g]==1]; ge=D[te_,g]==1
            if ge.any():
                m=LogisticRegression(C=1e6,max_iter=5000).fit(x[gt],y[gt]); out[ge]=m.predict_proba(x[te_][ge])[:,1]
        return out
def cv_pred(mode,attr,seed):
    P=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(x_:=np.zeros(len(y)),y):
        P[te]=fit_pred(tr,te,mode,attr)
    return np.clip(P,0.0003,0.9999)
def cil_stats(P,attr):
    c=[]
    for g in sorted(df[attr].unique()):
        m=(df[attr]==g).values
        if m.sum()>=30: c.append(P[m].mean()-y[m].mean())
    c=np.array(c); return np.abs(c).max(),np.abs(c).mean()
rows=[]
for attr in ATTR:
    for mode in ["platt","iso","offset","offset_shrunk","group_platt"]:
        acc=[]
        for seed in range(10):
            P=cv_pred(mode,attr,seed); mx,mn=cil_stats(P,attr)
            acc.append((brier_score_loss(y,P),log_loss(y,np.clip(P,1e-6,1-1e-6)),mx,mn))
        a=np.array(acc)
        rows.append(dict(attribute=attr,recalibrator=mode,brier=a[:,0].mean(),brier_sd=a[:,0].std(),logloss=a[:,1].mean(),
                         max_abs_cil=a[:,2].mean(),max_abs_cil_sd=a[:,2].std(),mean_abs_cil=a[:,3].mean()))
RC=pd.DataFrame(rows); RC.to_csv(OUT+"a2_recalibrator_by_group.csv",index=False)
print(RC.round(4).to_string(index=False))
