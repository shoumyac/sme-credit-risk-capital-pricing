from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"figure.dpi":200,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
F=str(ROOT/"paper"/"figures")+"/"
c1,c2,c3="#c0392b","#2980b9","#7f8c8d"
# F1 reliability + atoms
fig,ax=plt.subplots(1,2,figsize=(8.6,3.6))
def rel(P,lab,col,m="o"):
    q=np.quantile(P,np.linspace(0,1,11)); idx=np.clip(np.searchsorted(q,P,side="right")-1,0,9)
    xs=[P[idx==b].mean() for b in range(10) if (idx==b).any()]; ys=[y[idx==b].mean() for b in range(10) if (idx==b).any()]
    ax[0].plot(xs,ys,m+"-",ms=3.5,lw=1.2,color=col,label=lab)
ax[0].plot([0,1],[0,1],"k--",lw=.8); rel(raw,"Balanced-weight score",c1); rel(PD_platt,"Platt recalibrated",c2,"s"); rel(PD_iso,"Isotonic (18 atoms)",c3,"^")
ax[0].set_xlabel("Mean predicted probability (decile bins)"); ax[0].set_ylabel("Observed default rate"); ax[0].legend(frameon=False,fontsize=7.5); ax[0].set_title("(a) Reliability",fontsize=9.5)
b=np.linspace(0,1,41); ax[1].hist(PD_iso,b,alpha=.6,color=c1,label="Isotonic"); ax[1].hist(PD_platt,b,alpha=.6,color=c2,label="Platt")
ax[1].set_xlabel("Recalibrated PD"); ax[1].set_ylabel("Applicants"); ax[1].legend(frameon=False,fontsize=7.5); ax[1].set_title("(b) Isotonic: 343 borrowers at 3 bp",fontsize=9.5)
fig.savefig(F+"fig1_calibration.png"); plt.close(fig)
# F2 forest
T=pd.read_csv(OUT+"a1_subgroup_bootstrap.csv"); fig,ax=plt.subplots(1,3,figsize=(10.4,4.4),sharey=True)
labs=[f"{g} (n={n})" for g,n in zip(T[T.model=="global Platt"].group,T[T.model=="global Platt"].n)]; yy=np.arange(len(labs))[::-1]
for k,(mod,col) in enumerate((("raw balanced LR",c1),("global Platt",c2))):
    t=T[T.model==mod]; ax[0].errorbar(100*t.cil,yy+(0.15 if k==0 else -0.15),xerr=[100*(t.cil-t.cil_lo),100*(t.cil_hi-t.cil)],fmt="o",ms=3,color=col,lw=1,label=mod.replace("raw balanced LR","Balanced-weight score").replace("global Platt","Platt"))
ax[0].axvline(0,color="k",lw=.7); ax[0].set_xlabel("Predicted minus observed default rate (pp)"); ax[0].legend(frameon=False,fontsize=7.5); ax[0].set_yticks(yy); ax[0].set_yticklabels(labs,fontsize=7); ax[0].set_title("(a) Calibration in the large",fontsize=9.5)
t=T[T.model=="global Platt"]; ax[1].errorbar(t.slope,yy,xerr=[t.slope-t.slope_lo,t.slope_hi-t.slope],fmt="o",ms=3,color=c2,lw=1); ax[1].axvline(1,color="k",lw=.7); ax[1].set_xlabel("Calibration slope (Platt)"); ax[1].set_title("(b) Calibration slope",fontsize=9.5)
ax[2].errorbar(t.auc,yy,xerr=[t.auc-t.auc_lo,t.auc_hi-t.auc],fmt="o",ms=3,color=c3,lw=1); ax[2].axvline(0.899,color="k",lw=.7,ls=":"); ax[2].set_xlabel("ROC AUC"); ax[2].set_title("(c) Discrimination",fontsize=9.5)
fig.savefig(F+"fig2_subgroup_forest.png"); plt.close(fig)
# F3 K and rate vs PD
K=basel_K(PD_platt); r=rate(PD_platt,K=K); o=np.argsort(PD_platt)
fig,ax=plt.subplots(1,2,figsize=(8.6,3.5))
ax[0].scatter(100*PD_platt,100*K,s=3,color=c2,alpha=.5); ax[0].set_xlabel("Platt PD (%)"); ax[0].set_ylabel("Capital requirement K (% of EAD)"); ax[0].set_title("(a) Capital requirement",fontsize=9.5)
ax[1].scatter(100*PD_platt,100*r,s=3,color=c2,alpha=.5); ax[1].axhline(22,color=c1,ls="--",lw=1); ax[1].set_ylim(0,80); ax[1].set_xlabel("Platt PD (%)"); ax[1].set_ylabel("RAROC rate (%)"); ax[1].text(1,24,"22% ceiling",color=c1,fontsize=8); ax[1].set_title("(b) RAROC-consistent rate",fontsize=9.5)
fig.savefig(F+"fig3_capital_rate.png"); plt.close(fig)
# F4 sensitivity
SG=pd.read_csv(OUT+"b_sens_grid.csv"); CR=pd.read_csv(OUT+"b_cap_rf.csv")
fig,ax=plt.subplots(1,2,figsize=(8.6,3.5)); cols={0.30:"#27ae60",0.45:c2,0.60:"#e67e22",0.75:c1}
for l,s in SG.groupby("LGD"): ax[0].plot(100*s.hurdle,s.viable,color=cols[l],lw=1.5,label=f"LGD {l:.2f}")
ax[0].set_xlabel("RAROC hurdle (%)"); ax[0].set_ylabel("Share priceable under the cap (%)"); ax[0].legend(frameon=False,fontsize=7.5); ax[0].set_title("(a) LGD and hurdle",fontsize=9.5)
c=CR[CR.kind=="cap"]; ax[1].plot(100*c.value,c.viable,"o-",color=c2,ms=4); ax[1].set_xlabel("Rate ceiling (%)"); ax[1].set_ylabel("Share priceable (%)"); ax[1].set_title("(b) The assumed ceiling",fontsize=9.5)
fig.savefig(F+"fig4_sensitivity.png"); plt.close(fig)
# F5 guarantee
GU=pd.read_csv(OUT+"b_guarantee.csv").dropna(subset=["coverage"]); fig,ax=plt.subplots(1,2,figsize=(8.6,3.5))
ax[0].plot(GU.coverage,GU.viable,"o-",color=c2); ax[0].set_xlabel("Guarantee coverage (% of loss)"); ax[0].set_ylabel("Share priceable (%)"); ax[0].set_title("(a) Coverage restores access",fontsize=9.5)
g=GU[GU.coverage>0]; ax[1].plot(g.payout,g.incr,"o-",color="#27ae60")
for _,q in g.iterrows(): ax[1].annotate(f"{q.coverage:.0f}%",(q.payout,q.incr),fontsize=7,xytext=(4,-8),textcoords="offset points")
ax[1].set_xlabel("Expected fiscal cost (BDT m)"); ax[1].set_ylabel("Incremental credit (BDT m)"); ax[1].set_title("(b) Credit per unit of expected cost",fontsize=9.5)
fig.savefig(F+"fig5_guarantee.png"); plt.close(fig)
# F6 truncation
A4=pd.read_csv(OUT+"a4_truncation.csv"); fig,ax=plt.subplots(1,2,figsize=(8.6,3.5))
for pol,col,lab in (("scorecard",c1,"Historical scorecard cut"),("random",c3,"Random cut")):
    s=A4[(A4.policy==pol)&(A4.correction=="none")]; ax[0].plot(100*s.reject_share,100*s.cil_rej,"o-",color=col,label=lab,ms=4)
    ax[1].plot(100*s.reject_share,100*s.obs_default_rej,"o-",color=col,ms=4,label="Observed"); 
    ax[1].plot(100*s.reject_share,100*s.mean_pd_rej,"s--",color=col,ms=4,label="Model estimate")
ax[0].axhline(0,color="k",lw=.7); ax[0].set_xlabel("Share of applicants excluded from training (%)"); ax[0].set_ylabel("Predicted minus observed default, excluded group (pp)"); ax[0].legend(frameon=False,fontsize=7.5); ax[0].set_title("(a) Bias on the excluded group",fontsize=9.5)
ax[1].set_xlabel("Share excluded (%)"); ax[1].set_ylabel("Default rate, excluded group (%)"); ax[1].set_title("(b) Observed vs estimated",fontsize=9.5)
fig.savefig(F+"fig6_truncation.png"); plt.close(fig)
# F7 isotonic vs n
I=pd.read_csv(OUT+"a5_isotonic_vs_n.csv"); fig,ax=plt.subplots(1,2,figsize=(8.6,3.4))
ax[0].plot(I.n,I.iso_distinct,"o-",color=c1); ax[0].set_xscale("log"); ax[0].set_xlabel("Sample size (Taiwan card data, subsampled)"); ax[0].set_ylabel("Distinct isotonic PD values"); ax[0].set_title("(a) Atoms",fontsize=9.5)
ax[1].plot(I.n,I.viable_iso,"o-",color=c1,label="Isotonic"); ax[1].plot(I.n,I.viable_platt,"s-",color=c2,label="Platt"); ax[1].set_xscale("log"); ax[1].set_xlabel("Sample size"); ax[1].set_ylabel("Share priceable (%)"); ax[1].legend(frameon=False,fontsize=7.5); ax[1].set_title("(b) Priceable share",fontsize=9.5)
fig.savefig(F+"fig7_isotonic_n.png"); plt.close(fig)
print("ok")
