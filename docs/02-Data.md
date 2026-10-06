# Data

## Bangladeshi SME loan applications

- **Size:** 1,200 applications, 2022 to 2023. 303 missed at least one scheduled payment in the 12 months after disbursement (25.25%).
- **Source:** BASIC Bank Limited and Bangladesh Bank. The research was permitted by the Ministry of Finance of Bangladesh, and that permission was shown to the banks before the data were released.
- **Fields:** all columns, including the free-text descriptions and the outcome, were supplied by the banks.
- **Privacy:** the research extract has no personal identifiers.
- **Sharing:** the authors confirm the data can be shared publicly; it is released in `data/bangladesh_sme_1200_records.csv` under CC BY 4.0.
- **Column definitions:** [DATA_DICTIONARY.md](../data/DATA_DICTIONARY.md).

### Things to know before using it

- The label is **any missed payment**, which is wider than the Basel default (90 days past due or unlikely to pay). Section 8.4 of the paper shows why this matters.
- The column `missed_payment_last_12m` is the label in Yes/No form. Do not use it as a feature.
- Four free-text columns exist but this paper does not use them.
- There are no recoveries, so loss given default is assumed, not estimated.
- There are no rejected applicants, so selection bias cannot be corrected, only sized by experiment.
- Every borrower is below the EUR 5 million sales floor of the Basel SME size adjustment and below the EUR 1 million retail exposure threshold.

## Five public datasets used for replication

| Dataset | Country | n | Default rate | Source | Notes |
|---|---|---|---|---|---|
| German credit (Statlog) | Germany | 1,000 | 30% | OpenML `credit-g`; Hofmann (1994), UCI | Consumer loans |
| Default of credit card clients | Taiwan | 30,000 | 22.1% | OpenML; Yeh and Lien (2009) | Credit card default |
| Give Me Some Credit | United States | 150,000 | 6.7% | OpenML id 46929 | Consumer, serious delinquency |
| Polish companies bankruptcy (3 year horizon) | Poland | 10,503 | 4.7% | UCI id 365 | Listed firms, closest to SME |
| Australian credit approval | Australia | 690 | 44.5% | UCI id 143 | Anonymised features |

They are downloaded by `code/06_external_replication.py` into `data/external/` (not committed). They have no loss given default and some have no exposure amount, so the replication uses unit exposure and reports capital per unit of exposure. They test calibration and capital mechanics, not the Bangladeshi policy results.
