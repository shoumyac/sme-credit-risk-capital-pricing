# Methodology

Everything here is implemented in `code/common.py` (shared functions) and the numbered scripts.

## 1. Probability of default

- **Model:** logistic regression, L2 penalty (C = 1), class-balanced weights, on ten numeric and nine categorical features (see the data dictionary).
- **Out-of-fold scores:** stratified 5-fold cross validation, seed 42. Every borrower's score comes from a model that did not see that borrower.
- **Why class-balanced matters:** balancing moves the training prior to 50/50, so the score is not a default probability. This is the root of the first finding.

## 2. Recalibration

Four ways of turning the score into a PD are compared.

| Method | Definition |
|---|---|
| Raw score | Used as is (the naive route) |
| Platt scaling | Logistic regression of the outcome on logit(score). **Primary method.** |
| Isotonic regression | Monotone step function fitted to (score, outcome) |
| Analytic prior correction | odds x pi / (1 - pi), the closed form for a prior shift (Saerens et al. 2002) |

All PDs are clipped to [0.0003, 0.9999]. A **nested** version refits the recalibrator on four folds and applies it to the fifth, so the approval screen does not see its own labels.

## 3. Basel III IRB capital (per borrower)

Asset correlation (SME form):

R = 0.12 w + 0.24 (1 - w) - 0.04 (1 - (S - 5)/45), where w = (1 - exp(-50 PD)) / (1 - exp(-50)).

Maturity adjustment: MA = (1 + (M - 2.5) b) / (1 - 1.5 b), b = (0.11852 - 0.05478 ln PD)^2.

Capital per unit of exposure: K = LGD x [ Phi( (Phi^-1(PD) + sqrt(R) Phi^-1(0.999)) / sqrt(1 - R) ) - PD ] x MA.

RWA = 12.5 x K x EAD. Tier 1 capital = 6% of RWA.

Alternatives: the other-retail function (R = 0.03 w' + 0.16 (1 - w'), decay 35, no maturity adjustment) and the standardised approach (75% or 100% risk weight on 8%).

Note that the pipeline is a **foundation LGD, advanced maturity hybrid**: LGD is the foundation value of 45%, but maturity varies by borrower.

## 4. Risk-based price and the approval rule

Price that earns exactly the hurdle return on regulatory capital:

r = r_f + s_F + c_O + (PD x LGD + h x K) / (1 - PD)

| Parameter | Value | Where it comes from |
|---|---|---|
| r_f | 9% | Assumed, between 2022 and 2024 yields |
| s_F | 2% | Assumed funding spread |
| c_O | 3% | Assumed operating cost |
| h | 15% | Assumed hurdle (cost of equity) |
| LGD | 45% | Basel foundation value for senior unsecured |
| Ceiling | 22% | Assumed stylised rate cap between the bank cap and the microcredit cap |
| Flat rate | 18% | Benchmark flat rate |

A borrower is **priceable** if r is at most 22%. All of these are varied in sensitivity analyses.

Profit per borrower: [ (r - r_f - s_F - c_O)(1 - PD) - PD x LGD ] x EAD.

## 5. Diagnostics and uncertainty

- **Calibration:** Brier score, log loss, 5-bin expected calibration error, calibration slope and intercept, calibration in the large (CIL = mean prediction minus observed rate).
- **Headline uncertainty:** 1,000 bootstrap resamples over borrowers, with the recalibrator refitted in each (script 05).
- **Subgroups (18):** four sectors, seven divisions, urban/rural, five education levels. Percentile bootstrap intervals; z test of CIL with Holm correction; and a **parametric null**: outcomes are redrawn from the model's own probabilities (so the model is perfectly calibrated in every group) and the range across groups of slope, CIL and AUC is recorded, then compared with the observed range (script 02).
- **Capital and price bounds:** each group's PDs are shifted by the confidence limit of its CIL and capital and the median rate are recomputed (script 05).

## 6. Experiments that address data limitations

| Experiment | Idea | Script |
|---|---|---|
| Class weighting | Balanced vs unweighted training over five seeds, with and without recalibration | 04 |
| Truncation (selection) | Rank applicants by a crude historical scorecard, exclude the worst q%, train on the rest, evaluate on the excluded (whose labels we know). Random exclusion is the control; inverse-propensity weights are the standard repair | 04 |
| Thresholded fairness | Approval rate, approval among non-defaulters, impact ratio, with bootstrap intervals | 04 |
| Take-up and adverse selection | Take-up falls with the rate; accepted borrowers get riskier with the rate; grid over both | 04 |
| Downturn | Multiply default odds by 1.5, 2, 3; LGD 0.55 | 04 |
| Collateral-linked LGD | LGD 0.60 / 0.45 / 0.30 by collateral level | 04 |
| Per-group recalibration | Global Platt/isotonic vs group intercepts vs group-specific Platt, cross fitted | 03 |
| Default definition | Prior-shift correction assuming a share phi of missed payments qualify as Basel defaults | 01 |
| Guarantee | Cover a share of loss, or absorb only unexpected loss | 01 |
| Five-country replication | Same pipeline on five public datasets | 06, 07 |

## Seeds

All random processes use fixed seeds (cross validation seed 42; bootstrap and null generators have fixed seeds in each script), so a rerun reproduces the same numbers.
