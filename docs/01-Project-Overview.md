# Project overview

## The problem

Machine learning credit scoring is judged almost entirely on rank metrics such as ROC AUC. A lender, however, does not use a rank. It uses a **probability of default (PD)** to:

1. set **regulatory capital** under the Basel III internal ratings based (IRB) approach, a formula that is non linear in PD;
2. set a **price** so that the return on that capital clears a hurdle (RAROC);
3. **approve or reject** the loan, where a statutory rate ceiling can make high-risk loans impossible to price.

Because each step is non linear and depends on the one before, small errors in a probability are not small errors in an outcome.

## The idea

Follow one probability through all three steps, and change one upstream choice at a time to see what moves. The choices we vary are the ones that are usually treated as technical details:

- class weighting during training and how the score is recalibrated;
- the recalibrator (Platt scaling versus isotonic regression);
- whether subgroups are miscalibrated, and how much that could matter given small samples;
- selection bias, because models are trained only on loans that were approved;
- the regulatory treatment (corporate SME versus retail), the default definition, and loss given default;
- take-up, adverse selection and a downturn.

## What we found (short)

1. **Calibration is a first-order choice.** A class-balanced score is not a probability. Using it directly removes about 13 points of priceable share.
2. **Isotonic regression on small samples is an artefact generator.** 1,200 borrowers collapse onto 18 values; the capital formula amplifies the steps.
3. **Subgroup miscalibration is mostly noise at feasible sample sizes.** We show this with intervals and a sampling null, and bound the damage rather than assume it away. This reverses a common story, so we say so plainly.
4. **Selection bias is the bigger threat.** Models trained on screened applicants under-predict risk for the people they never saw.
5. **Regulatory treatment moves capital more than anything else we vary** (38% between corporate SME and retail).
6. **Under a binding rate ceiling, risk-based pricing is a screen.** The fall in default rate comes from refusing loans, not from the price. The cost lands on rural and agricultural borrowers.
7. **A guarantee that absorbs only unexpected loss** can restore access at no expected cost, in a stylised design.
8. **The calibration findings replicate** on data from Germany, Taiwan, the United States, Poland and Australia.

## Why the negative result matters

A natural and popular claim is that machine learning credit models are miscalibrated across demographic groups and that this drives unfair pricing. On these data that claim is not supported: the miscalibration is a uniform class-weighting artefact removed by one recalibration, and group dispersion is within sampling noise. In a 150,000-borrower US sample, slope differences across age bands are clearly detectable, so the conclusion is about statistical power as much as about models. Reporting this is part of the contribution.

## Who this is for

Credit risk researchers, regulators and central banks thinking about capital and rate ceilings, SME lenders in emerging markets, and anyone who wants a worked, reproducible example of carrying a model's output through to its economic consequence.
