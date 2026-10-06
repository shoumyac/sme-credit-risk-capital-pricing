from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
df=pd.read_csv(str(ROOT/"data"/"bangladesh_sme_1200_records.csv"),encoding="latin-1")
y=df["target_missed_payment"].values.astype(int)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
NUM=["business_age_years","employees_count","monthly_revenue_bdt","monthly_profit_bdt","loan_amount_bdt","preferred_tenure_months","cashflow_stability","repayment_confidence","prior_loan_count","days_payable_outstanding"]
CAT=["sector","legal_status","division","location_type","existing_loans","late_payment_history","loan_purpose","collateral_level","owner_education"]
def make_pipe(cw="balanced",C=1.0,seed=42):
    num_t=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())])
    cat_t=Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([("prep",ColumnTransformer([("num",num_t,NUM),("cat",cat_t,CAT)])),("clf",LogisticRegression(solver="lbfgs",max_iter=10000,class_weight=cw,C=C,random_state=seed))])
def oof(d,yy,cw="balanced",seed=42,k=5):
    o=np.zeros(len(d)); 
    for tr,te in StratifiedKFold(k,shuffle=True,random_state=seed).split(d,yy):
        p=make_pipe(cw,seed=seed).fit(d.iloc[tr],yy[tr]); o[te]=p.predict_proba(d.iloc[te])[:,1]
    return o
raw=oof(df,y)
OUT=str(ROOT/"results")+"/"
def lo(v):
    v=np.clip(v,1e-9,1-1e-9); return np.log(v/(1-v))
def inv(z): return 1/(1+np.exp(-z))
LGD,H,CAP,RF_,SF,CO=0.45,0.15,0.22,0.09,0.02,0.03
EAD=df["loan_amount_bdt"].values.astype(float)
M=np.clip(df["preferred_tenure_months"].values/12.0,1.0,5.0)
S=np.clip(12.0*df["monthly_revenue_bdt"].values/100.0/1e6,5.0,50.0)
def basel_K(p,lgd=LGD,S_=S,M_=M,treatment="corporate_sme"):
    p=np.asarray(p)
    x=(1-np.exp(-50*p))/(1-np.exp(-50.0))
    b=(0.11852-0.05478*np.log(p))**2
    if treatment=="corporate_sme":
        R=0.12*x+0.24*(1-x)-0.04*(1-(S_-5)/45); MA=(1+(M_-2.5)*b)/(1-1.5*b)
    elif treatment=="corporate_no_sme":
        R=0.12*x+0.24*(1-x); MA=(1+(M_-2.5)*b)/(1-1.5*b)
    elif treatment=="retail_other":
        xr=(1-np.exp(-35*p))/(1-np.exp(-35.0)); R=0.03*xr+0.16*(1-xr); MA=np.ones_like(p)
    K=lgd*(norm.cdf((norm.ppf(p)+np.sqrt(R)*norm.ppf(0.999))/np.sqrt(1-R))-p)*MA
    return np.maximum(K,0.0)
def rate(p,lgd=LGD,h=H,treatment="corporate_sme",K=None):
    if K is None: K=basel_K(p,lgd,treatment=treatment)
    return RF_+SF+CO+(p*lgd+h*K)/(1-p)
def platt_fit(s,yy):
    m=LogisticRegression(C=1e6,max_iter=10000).fit(lo(s).reshape(-1,1),yy); return m
def platt_pred(m,s): return np.clip(m.predict_proba(lo(s).reshape(-1,1))[:,1],0.0003,0.9999)
PD_platt=platt_pred(platt_fit(raw,y),raw)
PD_iso=np.clip(IsotonicRegression(out_of_bounds="clip").fit(raw,y).predict(raw),0.0003,0.9999)
GROUPS=["sector","division","location_type","owner_education"]
def group_masks(min_n=30):
    out=[]
    for a in GROUPS:
        for g in sorted(df[a].dropna().unique()):
            m=(df[a]==g).values
            if m.sum()>=min_n: out.append((a,g,m))
    return out
def slope_int(p,yy,iters=25):
    x=lo(p); X=np.c_[np.ones_like(x),x]; b=np.array([0.,1.])
    for _ in range(iters):
        mu=inv(X@b); W=mu*(1-mu)+1e-9
        g=X.T@(yy-mu); Hm=(X*W[:,None]).T@X+1e-9*np.eye(2)
        step=np.linalg.solve(Hm,g); b=b+step
        if np.abs(step).max()<1e-8: break
    return b[1],b[0]
