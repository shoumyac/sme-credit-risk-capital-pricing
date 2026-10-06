# FAQ

**Is this a review paper?** No. It is an empirical study with new analyses, tables and a replication.

**Why Platt scaling instead of isotonic regression?** With 1,200 observations isotonic regression is a step function with 18 steps, and 28.6% of borrowers land at the 3 bp floor. The capital formula amplifies that artefact. With much larger samples the artefact shrinks (see the Taiwan subsampling result).

**Does the paper claim machine learning credit models are unfair across groups?** No, and it says so. After recalibration there is no significant subgroup bias, and the spread across groups is within sampling noise. The approval gap between rural and urban borrowers follows from risk differences and the rate ceiling.

**Is the 22% ceiling a real rule?** It is a stylised value between the scheduled-bank cap and the microcredit cap that applied in the sample period. The paper varies it.

**Why unit exposure in the replication?** Several public datasets have no loan amount. Capital is reported per unit of exposure.

**Can I use the data?** Yes, under CC BY 4.0, with attribution. See `LICENSE-DATA-AND-PAPER`.

**How do I change a parameter such as LGD or the hurdle?** Edit the constants at the top of `code/common.py` and rerun.

**How do I cite it?** See `CITATION.cff`.
