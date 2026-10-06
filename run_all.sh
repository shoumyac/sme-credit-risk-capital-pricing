#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper. Needs internet once, for the five public data sets.
# Runtime: about 15 to 20 minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/code"
for s in 01_core_pipeline.py 02_subgroup_calibration.py 03_recalibration_by_group.py \
         04_weighting_selection_fairness_robustness.py 05_bootstrap_and_bounds.py \
         06_external_replication.py 07_isotonic_vs_sample_size.py 08_figures.py 09_build_paper.py; do
  echo "=== $s"; python "$s"
done
echo "Done. Compile the paper with: cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
