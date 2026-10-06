# Future work

Ordered roughly by how much they would improve the research.

## Highest value

1. **A real SME dataset from another country.** The single biggest gap. Even one more SME lender would move the paper from a case study to evidence. Candidate sources include central-bank SME credit registers, development-finance datasets and the World Bank Enterprise Surveys (which have financing questions but no defaults).
2. **Real recoveries.** Replace assumed LGD with estimated LGD, even from a small workout sample.
3. **Rejected-applicant outcomes or a policy change.** A natural experiment (for example a change in lending policy) would let selection bias be corrected rather than only sized. See Banasik and Crook (2007) for augmentation methods.

## Methodological extensions

4. **Other model families.** Carry gradient boosting and random forests through the same pipeline; ask whether calibration problems and capital pass-through differ.
5. **PD uncertainty.** Replace point PDs with conformal or Bayesian intervals and propagate them through the capital formula, giving capital and price intervals per borrower.
6. **Optimal recalibration for a non-linear downstream use.** Instead of calibrating for log loss, calibrate for the capital or price loss directly. This is the natural bridge to a machine learning paper (see below).
7. **Dynamic version.** Panel data would allow through-the-cycle PDs, procyclicality of capital and migration matrices.
8. **Welfare analysis.** Replace approval-rate fairness with a welfare measure of who is excluded and what the exclusion costs.

## Toward a machine learning venue

The cleanest theoretical question hidden in this study: **how does calibration error pass through a non-linear regulatory functional, and what recalibration minimises downstream loss?** A paper that states and proves bounds on capital error as a function of calibration error and the curvature of the IRB function, and tests them on the datasets here plus larger public ones, could be a candidate for a top ML venue (ICML or NeurIPS main track). It would need a real theorem and a stronger experimental section than this applied paper provides.

## Smaller practical items

- Add an interactive notebook that lets a reader change LGD, the hurdle and the ceiling.
- Add unit tests for `basel_K` against published Basel tables.
- Add a data card.
