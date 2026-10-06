# Data dictionary: `bangladesh_sme_1200_records.csv`

1,200 SME loan applications, Bangladesh, 2022 to 2023. One row per application. No missing values.
Source: BASIC Bank Limited and Bangladesh Bank, obtained for this research with permission from the Ministry of Finance of Bangladesh. The research extract contains no personal identifiers.

Descriptions below follow the column names as supplied; where a definition is not documented in the source we say so rather than guess.

| Column | Type | Values / range | Used as | Notes |
|---|---|---|---|---|
| `business_id` | text | SME0001 to SME1200 | identifier | Not a feature |
| `sector` | category | Retail 353, Manufacturing 337, Services 305, Agriculture 205 | feature, subgroup | |
| `business_age_years` | integer | 1 to 19 | feature | |
| `legal_status` | category | Sole 623, Limited 327, Partnership 250 | feature | |
| `division` | category | Dhaka 405, Chittagong 234, Khulna 161, Rajshahi 123, Barisal 102, Rangpur 90, Sylhet 85 | feature, subgroup | Administrative division |
| `location_type` | category | Urban 936, Rural 264 | feature, subgroup | |
| `employees_count` | integer | 2 to 49 | feature | |
| `monthly_revenue_bdt` | integer | 34,577 to 3,567,055 | feature; sales for the Basel size term | BDT per month |
| `monthly_profit_bdt` | integer | 5,198 to 1,337,403 | feature | BDT per month |
| `existing_loans` | category | Yes 830, No 370 | feature | |
| `prior_loan_count` | integer | 0 to 3 | feature | |
| `late_payment_history` | category | Never 793, Rarely 217, Sometimes 190 | feature | |
| `loan_purpose` | category | Equipment 301, Working capital 240, Expansion 231, Renovation 226, Inventory 202 | feature | |
| `loan_amount_bdt` | integer | 58,338 to 5,000,000 (mean about 2.16 million) | feature; exposure at default (EAD) | BDT |
| `preferred_tenure_months` | integer | 6 to 36 (mean 18.9) | feature; maturity M = months/12 clipped to 1 to 5 years | |
| `collateral_level` | category | High 431, Low 392, Moderate 377 | feature; collateral-linked LGD scenario | |
| `cashflow_stability` | integer | 1 to 5 | feature | Ordinal score as supplied |
| `repayment_confidence` | integer | 1 to 5 | feature | Ordinal score as supplied |
| `owner_education` | category | Higher Secondary 295, Secondary 264, Bachelor's 248, Primary 229, Master's 164 | feature, subgroup | |
| `days_payable_outstanding` | integer | 7 to 90 | feature | |
| `business_description`, `loan_purpose_description`, `repayment_plan_description`, `risk_factors_description` | text | free text | **not used** in this paper | Used only in earlier companion work on text features |
| `missed_payment_last_12m` | category | Yes 303, No 897 | outcome | Identical to `target_missed_payment`. **Never use as a feature (it is the label).** |
| `target_missed_payment` | 0/1 | 1 = missed any scheduled payment in the 12 months after disbursement (303 of 1,200, 25.25%) | **label** | Wider than the Basel default (90 days past due) |
| `split` | category | train 960, test 240 | not used | Split from earlier work. This paper uses stratified 5-fold cross validation instead |

## Features used by the model

Ten numeric (`business_age_years`, `employees_count`, `monthly_revenue_bdt`, `monthly_profit_bdt`, `loan_amount_bdt`, `preferred_tenure_months`, `cashflow_stability`, `repayment_confidence`, `prior_loan_count`, `days_payable_outstanding`) and nine categorical (`sector`, `legal_status`, `division`, `location_type`, `existing_loans`, `late_payment_history`, `loan_purpose`, `collateral_level`, `owner_education`). The code is in `code/common.py` (`NUM`, `CAT`).

## Derived inputs to the Basel formulas

| Input | Definition |
|---|---|
| EAD | `loan_amount_bdt` |
| M | `preferred_tenure_months / 12`, clipped to [1, 5] |
| S (annual sales, EUR million) | `12 * monthly_revenue_bdt / 100 / 1e6` (100 BDT per EUR), clipped to [5, 50]. Every borrower is below 5, so every borrower receives S = 5 |

## Reading the file

Plain CSV. If a tool complains about encoding, read it with `encoding="latin-1"` (the code does this).
