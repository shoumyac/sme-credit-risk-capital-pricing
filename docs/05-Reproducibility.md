# Reproducibility

## Requirements

Python 3.10 or later; packages in `requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib). No GPU. A LaTeX installation (pdflatex, bibtex) is needed only to rebuild the PDF. Internet access is needed once, to download the five public datasets.

## Run

```bash
pip install -r requirements.txt
bash run_all.sh
```

Or run the scripts in `code/` in numerical order, from inside `code/`. Typical runtime is 15 to 20 minutes; most of it is the 150,000-borrower US dataset and the bootstrap loops.

## Order and dependencies

| Script | Needs | Writes |
|---|---|---|
| 01 | data | `b_*.csv`, `V.json` |
| 02 | data | `a1_*.csv` |
| 03 | data | `a2_recalibrator_by_group.csv` |
| 04 | data | `a3_*`, `a4_*`, `a5_fairness_*`, `a6_takeup`, `a7_downturn`, `a9_*` |
| 05 | 02 | `a8_headline_ci.csv`, `a2_capital_price_bound.csv` |
| 06 | internet | `a6_external5*.csv` |
| 07 | internet | `a5_isotonic_vs_n.csv` |
| 08 | 01 to 07 | `paper/figures/*.png` |
| 09 | 01 to 07 | `paper/main.tex`, `X.json` |

## Determinism

Fixed seeds throughout (see [Methodology](03-Methodology.md)). Small differences in the last digit can appear across scikit-learn versions because of optimiser tolerances. The paper was produced with Python 3.13 and scikit-learn 1.7.

## Troubleshooting

- **A script cannot find `common`:** run it from inside `code/`.
- **Encoding error reading the CSV:** the code reads with `encoding="latin-1"`; do the same in your own scripts.
- **OpenML or UCI download fails:** check your connection and rerun; downloaded files are cached under `data/external/`.
- **LaTeX complains about missing packages:** install `adjustbox`, `natbib`, `booktabs`, `caption`, `enumitem`, `titlesec`.
- **Results differ from the paper:** check you ran scripts 01 to 07 before 09.

## Using the data in your own work

Load with `pandas.read_csv("data/bangladesh_sme_1200_records.csv", encoding="latin-1")`. The label is `target_missed_payment`. Use `code/common.py` for the Basel and pricing functions: `basel_K(pd, lgd, S, M, treatment)` and `rate(pd, ...)`.
