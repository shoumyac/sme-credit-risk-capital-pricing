from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.io import arff
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import external_helpers as H
import common as C
rng=np.random.default_rng(3)
EXT=ROOT/"data"/"external"
def ensure_uci():
    """Download the two UCI data sets (Polish bankruptcy, Australian credit) if absent."""
    import urllib.request, zipfile, io
    EXT.mkdir(parents=True,exist_ok=True)
    for flag,url in (("3year.arff","https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"),
                     ("australian.dat","https://archive.ics.uci.edu/static/public/143/statlog+australian+credit+approval.zip")):
        if not (EXT/flag).exists():
            zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(EXT)
def load(name):
    ensure_uci()
    if name=="Germany (German credit)":
        d=fetch_openml(name="credit-g",version=1,as_frame=True,data_home=str(ROOT/"data"/"external")); X=d.data.copy(); y=(d.target=="bad").astype(int).values
        X["age_band"]=pd.cut(X.age,[0,29,44,200],labels=["<30","30-44","45+"]).astype(str); return X,y,["age_band","personal_status","housing","foreign_worker"],"retail_other"
    if name=="Taiwan (card default)":
        d=fetch_openml(name="default-of-credit-card-clients",version=1,as_frame=True,data_home=str(ROOT/"data"/"external")); X=d.data.copy(); y=d.target.astype(int).values
        X["sex"]=X.x2.astype(int).astype(str); X["edu"]=X.x3.astype(int).astype(str); X["age_band"]=pd.cut(X.x5.astype(float),[0,29,39,49,200],labels=["<30","30-39","40-49","50+"]).astype(str); return X,y,["sex","edu","age_band"],"retail_other"
    if name=="United States (Give Me Some Credit)":
        d=fetch_openml(data_id=46929,as_frame=True,data_home=str(ROOT/"data"/"external")); X=d.data.copy(); y=(d.target=="Yes").astype(int).values
        age=[c for c in X.columns if "age" in c.lower()][0]; X["age_band"]=pd.cut(X[age].astype(float),[0,34,49,64,200],labels=["<35","35-49","50-64","65+"]).astype(str); return X,y,["age_band"],"retail_other"
    if name=="Poland (listed firms, bankruptcy)":
        a,_=arff.loadarff(str(ROOT/"data"/"external"/"3year.arff")); D=pd.DataFrame(a); y=D["class"].str.decode("utf-8").astype(int).values if D["class"].dtype==object else D["class"].astype(int).values
        X=D.drop(columns="class").astype(float); X.columns=[f"f{i}" for i in range(X.shape[1])]; X=X.clip(X.quantile(.01),X.quantile(.99),axis=1); return X,y,[],"corporate_no_sme"
    if name=="Australia (credit approval)":
        D=pd.read_csv(str(ROOT/"data"/"external"/"australian.dat"),sep=r"\s+",header=None); y=D.iloc[:,14].astype(int).values; X=D.iloc[:,:14].copy(); X.columns=[f"a{i}" for i in range(14)]
        for c in (0,3,4,5,11): X[f"a{c}"]=X[f"a{c}"].astype(str)
        return X,y,[],"retail_other"
def main():
    rows=[]; sub=[]; nul=[]
    for name in ("Germany (German credit)","Taiwan (card default)","United States (Give Me Some Credit)","Poland (listed firms, bankruptcy)","Australia (credit approval)"):
        X,y,G,tr=load(name); print(name,X.shape,round(y.mean(),4),flush=True)
        ead=np.ones(len(y))
        for cw in ("balanced",None):
            o=H.get_oof(X,y,cw); s=C.lo(o); pl=LogisticRegression(C=1e6,max_iter=5000).fit(s.reshape(-1,1),y); Pp=np.clip(pl.predict_proba(s.reshape(-1,1))[:,1],0.0003,0.9999)
            Pi=np.clip(IsotonicRegression(out_of_bounds="clip").fit(o,y).predict(o),0.0003,0.9999)
            for v,P in (("raw as PD",np.clip(o,0.0003,0.9999)),("Platt",Pp),("isotonic",Pi)):
                r=H.metr(P,y,ead,tr); r.update(dataset=name,n=len(y),prevalence=100*y.mean(),weighting=cw or "none",variant=v,auc=roc_auc_score(y,P),ece=H.ece5(P,y)); rows.append(r)
            if cw=="balanced" and G:
                gl=[]
                for a in G:
                    for g in sorted(set(map(str,X[a].tolist()))):
                        m=(np.array([str(v) for v in X[a].tolist()])==g)
                        if m.sum()>=50 and 0<y[m].sum()<m.sum(): gl.append((a,g,m))
                st=lambda P,yy,m:(C.slope_int(P[m],yy[m])[0],P[m].mean()-yy[m].mean())
                ob=np.array([st(Pp,y,m) for a,g,m in gl])
                for (a,g,m),(sl,ci) in zip(gl,ob): sub.append(dict(dataset=name,attribute=a,group=g,n=int(m.sum()),slope=sl,cil=ci))
                nr=[];nc=[]
                for _ in range(200):
                    ys=(rng.random(len(y))<Pp).astype(int); z=np.array([st(Pp,ys,m) for a,g,m in gl]); nr.append(np.ptp(z[:,0])); nc.append(np.ptp(z[:,1]))
                nul.append(dict(dataset=name,n_groups=len(gl),slope_range=np.ptp(ob[:,0]),slope_null_p95=np.percentile(nr,95),p_slope=(np.array(nr)>=np.ptp(ob[:,0])).mean(),cil_range=np.ptp(ob[:,1]),cil_null_p95=np.percentile(nc,95),p_cil=(np.array(nc)>=np.ptp(ob[:,1])).mean()))
    E=pd.DataFrame(rows); E.to_csv(C.OUT+"a6_external5.csv",index=False); pd.DataFrame(sub).to_csv(C.OUT+"a6_external5_subgroups.csv",index=False); N=pd.DataFrame(nul); N.to_csv(C.OUT+"a6_external5_null.csv",index=False)
    pd.set_option("display.width",250); print(E.round(4).to_string(index=False)); print(N.round(4).to_string(index=False))
if __name__=='__main__': main()
