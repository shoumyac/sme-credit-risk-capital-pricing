# Glossary

| Term | Meaning |
|---|---|
| **PD** | Probability of default, the lender's estimate that a borrower will default in the next 12 months |
| **LGD** | Loss given default, the share of the exposure lost if the borrower defaults (45% foundation value) |
| **EAD** | Exposure at default, the amount owed at default (here the loan amount) |
| **M** | Effective maturity in years |
| **IRB** | Internal ratings based approach, Basel's method for banks to compute capital from their own PDs |
| **K** | Capital requirement per unit of exposure from the Basel formula |
| **RWA** | Risk-weighted assets, 12.5 x K x EAD |
| **Tier 1 capital** | The highest-quality capital; here 6% of RWA |
| **Expected loss (EL)** | PD x LGD, the average loss that should be priced in |
| **Unexpected loss (UL)** | Loss beyond the average, what capital is held against |
| **RAROC** | Risk-adjusted return on capital; a loan is acceptable if profit divided by capital exceeds a hurdle |
| **Hurdle rate** | The minimum acceptable RAROC (15% here) |
| **Rate ceiling / cap** | A statutory maximum interest rate (assumed 22%) |
| **Priceable / viable** | A borrower whose RAROC-consistent rate is at or below the ceiling |
| **Calibration** | Predicted probabilities match observed frequencies |
| **Calibration in the large (CIL)** | Mean prediction minus observed rate |
| **Calibration slope** | Slope of a logistic regression of outcome on logit(prediction); 1 is ideal |
| **ECE** | Expected calibration error, average gap between predicted and observed in bins |
| **Brier score** | Mean squared error of the probabilities |
| **Platt scaling** | Logistic recalibration of a score |
| **Isotonic regression** | Monotone step-function recalibration |
| **Class weighting / balancing** | Training with weights that make both classes count equally; distorts probabilities |
| **Prior shift** | Change in the default rate between training and use |
| **Reject inference** | Methods to correct for training only on approved applicants |
| **IPW** | Inverse propensity weighting |
| **Truncation experiment** | Exclude part of a labelled sample from training and test on it, to measure selection bias |
| **Impact ratio** | Lowest group approval rate divided by highest; 0.8 is a common benchmark |
| **Adverse selection** | Higher prices attract riskier borrowers |
| **Partial credit guarantee** | A public guarantee covering part of the lender's loss |
| **BDT** | Bangladeshi taka |
