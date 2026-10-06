# Limitations and open problems

Status: **Done** = analysed and in the paper; **Partly** = bounded or sized but not removable with these data; **Open** = needs new data or work.

| # | Limitation (from the literature) | Status | What was done / what remains |
|---|---|---|---|
| 1 | Calibration rarely reported, or only pooled | Done | Full suite: reliability, Brier, ECE, slope, CIL |
| 2 | Subgroup calibration noisy at small n | Done | Bootstrap CIs, Holm tests, sampling null for 18 subgroups |
| 3 | Class weighting and prior shift studied outside credit | Done | Five seeds, analytic prior correction, prevalence grid |
| 4 | Per-group recalibration never compared | Done | Global vs group intercepts vs group-specific Platt |
| 5 | Fairness measured on thresholded decisions | Done | Approval, approval among non-defaulters, impact ratio |
| 6 | Reject inference / selection bias ignored | Partly | Truncation experiment sizes the bias; IPW does not repair it. True rejected outcomes unavailable |
| 7 | Capital and price effects only in aggregate | Done | Group-level bounds |
| 8 | Loss given default assumed | Partly | Grid and collateral-linked LGD. Needs real recoveries |
| 9 | No take-up or adverse selection | Partly | Parametric grid; parameters are assumed |
| 10 | Static analysis | Partly | Downturn scenarios. Needs panel or vintage data |
| 11 | Retail vs corporate not linked to pricing | Done | Treatment table includes rate and priceable share |
| 12 | Default definition softer than Basel | Partly | Sensitivity over the qualifying share. True 90 dpd label not observed |
| 13 | Single dataset | Partly | Five-country replication; mostly consumer loans |
| 14 | Assumed ceiling and funding rate | Done | Sensitivity |
| 15 | No uncertainty on headline numbers | Done | Bootstrap CIs |
| 16 | Isotonic artefact at small n | Done | Shown to shrink with n |
| 17 | Other model families (XGBoost, random forest) | Open | Only logistic regression carried through the pipeline |
| 18 | Point PD instead of intervals | Open | Conformal or Bayesian PD uncertainty |
| 19 | Real SME data in another country | Open | The strongest possible extension |
| 20 | Real recoveries, rejected-applicant outcomes, time dimension | Open | Needs bank data |

## Things a careful reader should keep in mind

- The label is any missed payment, not a Basel default.
- The 22% ceiling, the 9% risk-free rate, the 15% hurdle and the demand curve are stylised.
- The guarantee result is a stylised bound: it assumes the guarantor bears tail risk at no premium and ignores moral hazard.
- The in-sample approved-pool default rate benefits from the recalibrator seeing the labels; the nested figure is the one to quote.
- The replication datasets are not SME loans (Poland is listed firms), carry no LGD, and the cost parameters are borrowed from the Bangladeshi setting.
