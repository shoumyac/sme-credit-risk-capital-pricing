from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import re, json, numpy as np, pandas as pd
O=str(ROOT/"results")+"/"
V=json.load(open(O+"V.json"))
rd=lambda n:pd.read_csv(O+n)
def f(x,d=1):
    if pd.isna(x): return "--"
    return f"{x:,.{d}f}".replace("-","$-$") if x<0 else f"{x:,.{d}f}"
def ci(a,lo,hi,d=1,s=1): return f"{f(a*s,d)} [{f(lo*s,d)}, {f(hi*s,d)}]"
import re as _re
def esc(s): return _re.sub(r"(?<!\\)%",r"\\%",str(s))
def tab(cols,rows,align=None,head=None):
    align=align or "l"+"r"*(len(cols)-1)
    s="\\begin{tabular}{"+align+"}\n\\toprule\n"+" & ".join(cols)+" \\\\\n\\midrule\n"
    for r in rows:
        s+= ("\\midrule\n" if r=="MID" else " & ".join(esc(c) for c in r)+" \\\\\n")
    return "\\begin{adjustbox}{max width=\\textwidth}\n"+s+"\\bottomrule\n\\end{tabular}\n\\end{adjustbox}"
T={}
# T headline
RC=rd("b_recalibrators.csv")
rows=[]
for _,r in RC.iterrows():
    nm=r.tag
    if nm=="Raw balanced score": rows.append([ "Balanced-weight score",f(r.mean_pd,1),f(r.brier,3),f(r.auc,3),f(r.distinct,0),"--","--","--","--","--"]); continue
    rows.append([nm.replace("Isotonic nested","Isotonic, nested").replace("Platt nested","Platt, nested"),f(r.mean_pd,1),f(r.brier,3),f(r.auc,3),f(r.distinct,0),f(r.at_floor,0),f(r.mean_K,2),f(r.total_t1,1),f(r.viable,1),f(r.pool_default,2)])
T["recal"]=tab(["PD estimate","Mean PD (\\%)","Brier","AUC","Distinct PDs","At 3 bp","Mean $K$ (\\%)","Tier 1 (BDT m)","Priceable (\\%)","Default in pool (\\%)"],rows,"lrrrrrrrrr")
# class weighting
A3=rd("a3_class_weighting.csv"); rows=[]
for _,r in A3.sort_values(["weighting","variant"]).iterrows(): rows.append([r.weighting.capitalize() if r.weighting=="none" else "Balanced",r.variant.replace("raw as PD","Raw score used as PD").replace("analytic prior correction","Analytic prior correction").replace("Platt","Platt recalibration"),f(100*r.mean_pd,1),f(r.brier,3),f(r.ece,3),f(r.tier1_m,1),f(r.share_viable,1)])
T["cw"]=tab(["Training weights","Probability used","Mean PD (\\%)","Brier","ECE","Tier 1 (BDT m)","Priceable (\\%)"],rows,"llrrrrr")
PS=rd("a3_prevalence_shift.csv"); T["ps"]=tab(["Assumed default rate (\\%)","Mean PD (\\%)","Mean $K$ (\\%)","Tier 1 (BDT m)","Priceable (\\%)"],[[f(100*r.prevalence,1),f(100*r.mean_pd,1),f(r.mean_K,2),f(r.tier1_m,1),f(r.share_viable,1)] for _,r in PS.iterrows()])
# subgroup table
S=rd("a1_subgroup_bootstrap.csv"); R_=S[S.model=="raw balanced LR"].reset_index(drop=True); P_=S[S.model=="global Platt"].reset_index(drop=True); rows=[]; last=None
for i in range(len(R_)):
    r,p=R_.loc[i],P_.loc[i]
    if last and r.attribute!=last: rows.append("MID")
    last=r.attribute
    rows.append([f"{r.group}".replace("'","'"),int(r.n),f(100*r.obs,1),ci(r.cil,r.cil_lo,r.cil_hi,1,100),ci(p.cil,p.cil_lo,p.cil_hi,1,100),ci(p.slope,p.slope_lo,p.slope_hi,2),ci(p.auc,p.auc_lo,p.auc_hi,2)])
T["sub"]=tab(["Subgroup","$n$","Observed (\\%)","Raw CIL (pp)","Platt CIL (pp)","Platt slope","AUC"],rows,"lrrrrrr")
N=rd("a1_null_dispersion.csv"); rows=[]
for _,r in N.iterrows():
    nm=r.model.replace("raw balanced LR","Balanced-weight score").replace("global Platt","Platt")
    rows.append([nm,"Slope range",f(r.slope_range_obs,2),f(r.slope_range_null_median,2),f(r.slope_range_null_p95,2),f(r.p_slope,3)])
    rows.append(["","CIL range (pp)",f(100*r.cil_range_obs,1),f(100*r.cil_range_null_median,1),f(100*r.cil_range_null_p95,1),f(r.p_cil,3)])
    rows.append(["","AUC range",f(r.auc_range_obs,3),f(r.auc_range_null_median,3),f(r.auc_range_null_p95,3),f(r.p_auc,3)])
T["null"]=tab(["Scores","Statistic","Observed","Null median","Null 95th pct","$p$"],rows,"llrrrr")
RG=rd("a2_recalibrator_by_group.csv"); rows=[]
nm={"platt":"Global Platt","iso":"Global isotonic","offset":"Platt plus group intercept","offset_shrunk":"Same, shrunk (L2)","group_platt":"Group-specific Platt"}
for a in ["sector","division","location_type","owner_education"]:
    for k in ["platt","iso","offset","offset_shrunk","group_platt"]:
        r=RG[(RG.attribute==a)&(RG.recalibrator==k)].iloc[0]; rows.append([a.replace("_"," ").capitalize() if k=="platt" else "",nm[k],f(r.brier,4),f(r.logloss,3),f(100*r.max_abs_cil,2),f(100*r.mean_abs_cil,2)])
    rows.append("MID")
T["recgrp"]=tab(["Attribute","Recalibrator","Brier","Log loss","Max $|$CIL$|$ (pp)","Mean $|$CIL$|$ (pp)"],rows[:-1],"llrrrr")
B=rd("a2_capital_price_bound.csv"); rows=[]
for a in ["sector","division","location_type","owner_education"]:
    b=B[B.attribute==a]; up=b[b.bound=="CIL upper"]; lw=b[b.bound=="CIL lower"]
    rows.append([a.replace("_"," ").capitalize(),f"{f(lw.capital_misstatement_pct.min(),0)} to {f(lw.capital_misstatement_pct.max(),0)}",f"{f(up.capital_misstatement_pct.min(),0)} to {f(up.capital_misstatement_pct.max(),0)}",f"{f(lw.rate_diff_pp.min(),1)} to {f(lw.rate_diff_pp.max(),1)}",f"{f(up.rate_diff_pp.min(),1)} to {f(up.rate_diff_pp.max(),1)}"])
T["bound"]=tab(["Attribute","Capital error, PD under-stated (\\%)","Capital error, PD over-stated (\\%)","Rate error, under-stated (pp)","Rate error, over-stated (pp)"],rows,"lrrrr")
# headline CI
H=rd("a8_headline_ci.csv").set_index("Unnamed: 0"); lab={"auc":("ROC AUC",3),"mean_K":("Mean capital requirement $K$ (\\%)",2),"tier1_m":("Total Tier 1 capital (BDT m)",1),"share_viable":("Share priceable under the cap (\\%)",1),"default_in_pool":("Default rate in priceable pool (\\%)",2),"flat_profit_m":("Flat 18\\% regime profit (BDT m)",0),"median_rate":("Median RAROC rate (\\%)",1)}
T["ci"]=tab(["Quantity","Estimate","95\\% bootstrap interval"],[[lab[k][0],f(H.loc[k,"estimate"],lab[k][1]),f"[{f(H.loc[k,'lo'],lab[k][1])}, {f(H.loc[k,'hi'],lab[k][1])}]"] for k in lab],"lrr")
G=rd("b_groups.csv"); rows=[]; last=None
for _,r in G.iterrows():
    if last and r.attribute!=last: rows.append("MID")
    last=r.attribute; rows.append([r.group,int(r.n),f(r.pd,1),f(r.obs,1),f(r.K,2),f(r.t1_m,1),f(r.median_rate,1),f(r.p75_rate,1),f(r.viable,1)])
T["groups"]=tab(["Group","$n$","Mean PD (\\%)","Observed (\\%)","Mean $K$ (\\%)","Tier 1 (BDT m)","Median rate (\\%)","P75 rate (\\%)","Priceable (\\%)"],rows,"lrrrrrrrr")
RGm=rd("b_regime.csv"); T["regime"]=tab(["Regime","Approval (\\%)","Default in pool (\\%)","Volume (BDT m)","Expected loss (BDT m)","Profit (BDT m)"],[[r.regime,f(r.approval,1),f(r.default,2),f(r.volume_m,0),f(r.el_m,1),f(r.profit_m,1)] for _,r in RGm.iterrows()],"lrrrrr")
SG=rd("b_sens_grid.csv"); hs=[0.05,0.10,0.15,0.20,0.25]; rows=[]
for l in (0.30,0.45,0.60,0.75): rows.append([f(l,2)]+[f(SG[(SG.LGD==l)&(np.isclose(SG.hurdle,h))].viable.iloc[0],1) for h in hs])
T["sens"]=tab(["LGD"]+[f"$h$ = {int(100*h)}\\%" for h in hs],rows,"lrrrrr")
CR=rd("b_cap_rf.csv"); T["caprf"]=tab(["Ceiling (\\%)","Priceable (\\%)","","Funding $r_f$ (\\%)","Priceable (\\%)"],[[f(100*a.value,0),f(a.viable,1),"",(f(100*b.value,0) if b is not None else ""),(f(b.viable,1) if b is not None else "")] for a,b in zip([r for _,r in CR[CR.kind=="cap"].iterrows()],[r for _,r in CR[CR.kind=="rf"].iterrows()]+[None]*3)],"rrrrr")
TR=rd("b_treatment.csv"); T["treat"]=tab(["Treatment","$K$ iso (\\%)","Tier 1 iso","Priceable iso (\\%)","$K$ Platt (\\%)","Tier 1 Platt","Priceable Platt (\\%)","Median rate Platt (\\%)"],[[r.treatment,f(r.K_iso,2),f(r.t1_iso,1),f(r.viable_iso,1),f(r.K_platt,2),f(r.t1_platt,1),f(r.viable_platt,1),f(r.rate_platt,1)] for _,r in TR.iterrows()],"lrrrrrrr")
DD=rd("b_defdef.csv"); T["dd"]=tab(["Basel-qualifying share $\\phi$","Implied default rate (\\%)","Mean $K$ (\\%)","Tier 1 (BDT m)","Median rate (\\%)","Priceable (\\%)"],[[f(r.phi,2),f(r.rate,1),f(r.K,2),f(r.t1,1),f(r.median_rate,1),f(r.viable,1)] for _,r in DD.iterrows()])
A9=rd("a9_lgd_collateral.csv"); rows=[]
for _,r in A9.iterrows(): rows.append([r.lgd_scheme if r.collateral=="Low" else "",r.collateral,int(r.n),f(r.lgd,2),f(100*r.default_rate,1),f(r.mean_K,2),f(r.share_viable,1),f(r.median_rate,1)])
T["lgd"]=tab(["LGD scheme","Collateral","$n$","LGD","Default (\\%)","Mean $K$ (\\%)","Priceable (\\%)","Median rate (\\%)"],rows,"llrrrrrr")
A4=rd("a4_truncation.csv"); rows=[]
for pol,lab_ in (("scorecard","Scorecard cut"),("random","Random cut")):
    for q in (0.1,0.3,0.5):
        a=A4[(A4.policy==pol)&(A4.reject_share==q)&(A4.correction=="none")].iloc[0]; b=A4[(A4.policy==pol)&(A4.reject_share==q)&(A4.correction=="IPW")].iloc[0]
        rows.append([lab_ if q==0.1 else "",f"{int(100*q)}\\%",int(a.n_rej),f(100*a.obs_default_rej,1),f(100*a.mean_pd_rej,1),f(100*a.cil_rej,1),f(a.auc_rej,3),f(100*b.cil_rej,1)])
T["trunc"]=tab(["Exclusion rule","Excluded","$n$ excluded","Observed (\\%)","Predicted (\\%)","Bias (pp)","AUC","Bias with IPW (pp)"],rows,"llrrrrrr")
A6=rd("a6_takeup.csv"); T["take"]=tab(["Take-up slope $k$","Adverse selection $\\gamma$","Flat profit (BDT m)","RAROC profit (BDT m)","RAROC volume (BDT m)","Default, RAROC take-up (\\%)"],[[f(r.takeup_slope,2),f(r.adverse_gamma,2),f(r.flat_profit,1),f(r.raroc_profit,1),f(r.raroc_volume,0),f(100*r.raroc_pd,2)] for _,r in A6.iterrows()],"rrrrrr")
A7=rd("a7_downturn.csv"); T["down"]=tab(["PD odds multiplier","LGD","Mean PD (\\%)","Tier 1 (BDT m)","Priceable (\\%)","Median rate (\\%)"],[[f(r.odds_multiplier,1),f(r.lgd,2),f(100*r.mean_pd,1),f(r.tier1_m,1),f(r.share_viable,1),f(r.median_rate,1)] for _,r in A7.iterrows()])
FG=rd("a5_fairness_gaps.csv"); T["fair"]=tab(["Attribute","Approval gap (pp)","Gap, good borrowers (pp)","Impact ratio"],[[r.attribute.replace("_"," ").capitalize(),ci(r.approval_gap,r.approval_gap_lo,r.approval_gap_hi,1,100),ci(r.good_gap,r.good_gap_lo,r.good_gap_hi,1,100),ci(r.impact_ratio,r.ratio_lo,r.ratio_hi,2)] for _,r in FG.iterrows()],"lrrr")
GU=rd("b_guarantee.csv"); T["guar"]=tab(["Design","Mean $K$ (\\%)","Median rate (\\%)","Priceable (\\%)","Credit (BDT m)","Incremental (BDT m)","Expected outlay (BDT m)"],[[("Unexpected-loss only" if pd.isna(r.coverage) else f"Coverage {int(r.coverage)}\\%"),f(r.K,2),f(r.median_rate,1),f(r.viable,1),f(r.credit,0),f(r.incr,0),f(r.payout,1)] for _,r in GU.iterrows()],"lrrrrrr")
import os
if os.path.exists(O+"a6_external5.csv"):
    E=rd("a6_external5.csv"); rows=[]
    for d,g in E.groupby("dataset",sort=False):
        b=g[(g.weighting=="balanced")&(g.variant=="raw as PD")].iloc[0]; p=g[(g.weighting=="balanced")&(g.variant=="Platt")].iloc[0]; i=g[(g.weighting=="balanced")&(g.variant=="isotonic")].iloc[0]; u=g[(g.weighting=="none")&(g.variant=="raw as PD")].iloc[0]
        rows.append([d,f(b.n,0),f(b.prevalence,1),f(100*b.mean_pd,1),f(100*u.mean_pd,1),f(100*(1-p.brier/b.brier),1),f(b.share_viable,1),f(p.share_viable,1),f(i.distinct,0),f(100*(i.tier1_per_unit/p.tier1_per_unit-1),1)])
    T["ext"]=tab(["Dataset","$n$","Default (\\%)","Balanced raw PD (\\%)","Unweighted raw PD (\\%)","Brier gain, Platt (\\%)","Priceable, raw (\\%)","Priceable, Platt (\\%)"],[r[:8] for r in rows],"lrrrrrrr")
    T["ext2"]=tab(["Dataset","Isotonic atoms","Tier 1, isotonic vs Platt (\\%)"],[[r[0],r[8],r[9]] for r in rows],"lrr")
    Nn=rd("a6_external5_null.csv"); T["extnull"]=tab(["Dataset","Subgroups","Slope range","Null 95th pct","$p$","CIL range (pp)","Null 95th pct (pp)","$p$"],[[r.dataset,int(r.n_groups),f(r.slope_range,2),f(r.slope_null_p95,2),f(r.p_slope,3),f(100*r.cil_range,1),f(100*r.cil_null_p95,1),f(r.p_cil,3)] for _,r in Nn.iterrows()],"lrrrrrrr")
I=rd("a5_isotonic_vs_n.csv"); T["isn"]=tab(["Sample size","Isotonic atoms","At floor (\\%)","Tier 1 ratio, iso / Platt","Priceable iso (\\%)","Priceable Platt (\\%)"],[[f(r.n,0),f(r.iso_distinct,1),f(r.iso_at_floor_pct,2),f(r.tier1_ratio_iso_over_platt,3),f(r.viable_iso,1),f(r.viable_platt,1)] for _,r in I.iterrows()])
# extra scalars
X=dict(V)
x=lambda **k: X.update(k)
n_=rd("a1_null_dispersion.csv"); r0=n_.iloc[0]; r1=n_.iloc[1]
x(slope_rng_raw=r0.slope_range_obs,slope_null_raw=r0.slope_range_null_median,slope_p_raw=r0.p_slope,cil_rng_raw=100*r0.cil_range_obs,cil_null_raw=100*r0.cil_range_null_median,auc_rng=r0.auc_range_obs,auc_null=r0.auc_range_null_median,auc_p=r0.p_auc,slope_rng_p=r1.slope_range_obs,slope_p_p=r1.p_slope)
Sx=rd("a1_subgroup_bootstrap.csv"); rawS=Sx[Sx.model=="raw balanced LR"]; x(raw_cil_min=100*rawS.cil.min(),raw_cil_max=100*rawS.cil.max(),slope_min=Sx[Sx.model=="raw balanced LR"].slope.min(),slope_max=Sx[Sx.model=="raw balanced LR"].slope.max())
x(plat_sig=int((Sx[Sx.model=="global Platt"].cil_p_holm<0.05).sum()),raw_sig=int((rawS.cil_p_holm<0.05).sum()))
fg=rd("a5_fairness_gaps.csv").set_index("attribute"); x(rural_gap=100*fg.loc["location_type","approval_gap"],rural_lo=100*fg.loc["location_type","approval_gap_lo"],rural_hi=100*fg.loc["location_type","approval_gap_hi"],rural_ratio=fg.loc["location_type","impact_ratio"],rural_r_lo=fg.loc["location_type","ratio_lo"],rural_r_hi=fg.loc["location_type","ratio_hi"])
g_=rd("b_groups.csv").set_index("group"); x(rural_app=g_.loc["Rural","viable"],urban_app=g_.loc["Urban","viable"],rural_pd=g_.loc["Rural","pd"],urban_pd=g_.loc["Urban","pd"],rural_rate=g_.loc["Rural","median_rate"],urban_rate=g_.loc["Urban","median_rate"],agri_app=g_.loc["Agriculture","viable"],agri_rate=g_.loc["Agriculture","median_rate"])
cw=rd("a3_class_weighting.csv").set_index(["weighting","variant"]); x(cw_bal_raw=100*cw.loc[("balanced","raw as PD"),"mean_pd"],cw_none_raw=100*cw.loc[("none","raw as PD"),"mean_pd"],cw_bal_via=cw.loc[("balanced","raw as PD"),"share_viable"],cw_pl_via=cw.loc[("balanced","Platt"),"share_viable"],cw_none_via=cw.loc[("none","raw as PD"),"share_viable"],cw_bal_ece=cw.loc[("balanced","raw as PD"),"ece"],cw_pl_ece=cw.loc[("balanced","Platt"),"ece"],cw_ana_ece=cw.loc[("balanced","analytic prior correction"),"ece"],cw_ana_via=cw.loc[("balanced","analytic prior correction"),"share_viable"])
x(cw_bal_t1=cw.loc[("balanced","raw as PD"),"tier1_m"],cw_pl_t1=cw.loc[("balanced","Platt"),"tier1_m"])
a4=rd("a4_truncation.csv"); sc=a4[(a4.policy=="scorecard")&(a4.correction=="none")]; ra=a4[(a4.policy=="random")&(a4.correction=="none")]; ip=a4[(a4.policy=="scorecard")&(a4.correction=="IPW")]
x(sc_cil_min=100*sc.cil_rej.min(),sc_cil_max=100*sc.cil_rej.max(),ra_cil_abs=100*ra.cil_rej.abs().max(),sc_auc_min=sc.auc_rej.min(),sc_auc_max=sc.auc_rej.max(),ip_cil_min=100*ip.cil_rej.min(),ip_cil_max=100*ip.cil_rej.max())
a6=rd("a6_takeup.csv"); x(tk_lo=a6.raroc_profit.min(),tk_hi=a6.raroc_profit.max(),flat_tk=a6.flat_profit.iloc[0])
a7=rd("a7_downturn.csv"); x(dn_v2=a7[(a7.odds_multiplier==2)&(a7.lgd==0.45)].share_viable.iloc[0],dn_t2=a7[(a7.odds_multiplier==2)&(a7.lgd==0.45)].tier1_m.iloc[0],dn_t1=a7.tier1_m.iloc[0],dn_lgd_t=a7[(a7.odds_multiplier==1)&(a7.lgd==0.55)].tier1_m.iloc[0],dn_lgd_v=a7[(a7.odds_multiplier==1)&(a7.lgd==0.55)].share_viable.iloc[0],dn_v3=a7[(a7.odds_multiplier==3)&(a7.lgd==0.45)].share_viable.iloc[0],dn_rate3=a7[(a7.odds_multiplier==3)&(a7.lgd==0.45)].median_rate.iloc[0])
dd=rd("b_defdef.csv"); x(dd05=dd[dd.phi==0.5].viable.iloc[0]-dd[dd.phi==1.0].viable.iloc[0],dd_t05=dd[dd.phi==0.5].t1.iloc[0])
sg=rd("b_sens_grid.csv"); b=sg[(sg.LGD==0.45)&(np.isclose(sg.hurdle,0.15))].viable.iloc[0]; x(lgd30=sg[(sg.LGD==0.30)&(np.isclose(sg.hurdle,0.15))].viable.iloc[0]-b,lgd60=sg[(sg.LGD==0.60)&(np.isclose(sg.hurdle,0.15))].viable.iloc[0]-b,h25=sg[(sg.LGD==0.45)&(np.isclose(sg.hurdle,0.25))].viable.iloc[0]-b,h05=sg[(sg.LGD==0.45)&(np.isclose(sg.hurdle,0.05))].viable.iloc[0]-b)
cr=rd("b_cap_rf.csv"); x(cap18=cr[(cr.kind=="cap")&np.isclose(cr.value,.18)].viable.iloc[0],cap26=cr[(cr.kind=="cap")&np.isclose(cr.value,.26)].viable.iloc[0],rf7=cr[(cr.kind=="rf")&np.isclose(cr.value,.07)].viable.iloc[0],rf11=cr[(cr.kind=="rf")&np.isclose(cr.value,.11)].viable.iloc[0])
a9=rd("a9_lgd_collateral.csv"); x(lgd_low_v=a9[(a9.lgd_scheme=="collateral-linked (0.60/0.45/0.30)")&(a9.collateral=="Low")].share_viable.iloc[0],lgd_hi_v=a9[(a9.lgd_scheme=="collateral-linked (0.60/0.45/0.30)")&(a9.collateral=="High")].share_viable.iloc[0],lgd_all_v=a9[(a9.lgd_scheme=="collateral-linked (0.60/0.45/0.30)")&(a9.collateral=="ALL")].share_viable.iloc[0],lgd_all_t=a9[(a9.lgd_scheme=="collateral-linked (0.60/0.45/0.30)")&(a9.collateral=="ALL")].tier1_m.iloc[0],lgd_low_def=100*a9[a9.collateral=="Low"].default_rate.iloc[0],lgd_hi_def=100*a9[a9.collateral=="High"].default_rate.iloc[0],lgd_mod_def=100*a9[a9.collateral=="Moderate"].default_rate.iloc[0])
bd=rd("a2_capital_price_bound.csv"); x(bd_cap_lo=bd.capital_misstatement_pct.min(),bd_cap_hi=bd.capital_misstatement_pct.max(),bd_rate_lo=bd.rate_diff_pp.min(),bd_rate_hi=bd.rate_diff_pp.max())
h=H; x(ci_v_lo=h.loc["share_viable","lo"],ci_v_hi=h.loc["share_viable","hi"],ci_t_lo=h.loc["tier1_m","lo"],ci_t_hi=h.loc["tier1_m","hi"],ci_auc_lo=h.loc["auc","lo"],ci_auc_hi=h.loc["auc","hi"],ci_f_lo=-h.loc["flat_profit_m","hi"],ci_f_hi=-h.loc["flat_profit_m","lo"],ci_pool_lo=h.loc["default_in_pool","lo"],ci_pool_hi=h.loc["default_in_pool","hi"])
x(flat_vs_screen=RGm.profit_m.iloc[3],screen_profit=RGm.profit_m.iloc[2])
x(cw_gap=X["cw_pl_via"]-X["cw_bal_via"])
x(guar_lo=GU.incr.iloc[1],guar_pay_lo=GU.payout.iloc[1])
iv=rd("a5_isotonic_vs_n.csv"); x(tw_atoms_500=iv.iso_distinct.iloc[0],tw_atoms_30k=iv.iso_distinct.iloc[-1],tw_viab_iso_500=iv.viable_iso.iloc[0],tw_viab_pl_500=iv.viable_platt.iloc[0])
if os.path.exists(O+"a6_external5.csv"):
    E=rd("a6_external5.csv"); bal=E[(E.weighting=="balanced")]
    bb=bal[bal.variant=="raw as PD"].set_index("dataset"); pp=bal[bal.variant=="Platt"].set_index("dataset"); ii=bal[bal.variant=="isotonic"].set_index("dataset")
    x(ext_bg_min=100*(1-pp.brier/bb.brier).min(),ext_bg_max=100*(1-pp.brier/bb.brier).max(),ext_over_min=(bb.mean_pd/(bb.prevalence/100)).min(),ext_over_max=(bb.mean_pd/(bb.prevalence/100)).max())
    x(ext_n=len(bb),ext_pol=bb.loc['Poland (listed firms, bankruptcy)','prevalence'])
    nn=rd('a6_external5_null.csv').set_index('dataset'); x(us_slope=nn.loc['United States (Give Me Some Credit)','slope_range'],us_null=nn.loc['United States (Give Me Some Credit)','slope_null_p95'])
json.dump({k:float(v) for k,v in X.items()},open(O+"X.json","w"),indent=1)
# fill template
src=open(str(ROOT/"paper"/"main_template.tex"),encoding="utf-8").read()
def sub(m):
    neg=m.group(1)=="-"; k=m.group(2); fmt=m.group(3)
    if k.startswith("T_"): return T[k[2:]]
    v=-X[k] if neg else X[k]
    s=format(v,fmt or ".1f")
    return s.replace("-","$-$")
out=re.sub(r"\{\{(-?)(\w+)(?::([^}]+))?\}\}",sub,src)
open(str(ROOT/"paper"/"main.tex"),"w",encoding="utf-8").write(out); print("built",len(out))
