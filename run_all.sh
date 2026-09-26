#!/usr/bin/env bash
# Power Puff Girls -- business entity resolution pipeline
#
# End-to-end runner: trains on the train split, then produces
# output/candidate_pairs.tsv and output/matching_results.tsv for the test split.
#
# Usage:  bash run_all.sh [JOBS]
#   JOBS   number of parallel workers (default 16)
set -euo pipefail
JOBS="${1:-16}"
SRC=src
mkdir -p output artifacts

echo "=== [1/7] Exploratory data analysis ==="
python  -m $SRC.pipeline eda --split train --limit 20000

echo "=== [2/7] Prepare (normalize) train data ==="
python  -m $SRC.pipeline prepare --split train --jobs "$JOBS"

echo "=== [3/7] Ground-truth pairs ==="
python  -m $SRC.pipeline gt --split train

echo "=== [4/7] Blocking indexes + candidates (train) ==="
python  -m $SRC.pipeline index --split train
python  -m $SRC.pipeline block --split train --jobs "$JOBS"

echo "=== [5/7] Blocking recall ceiling (train) ==="
python  -m $SRC.pipeline recall --split train

echo "=== [6/7] Features -> train model -> tune threshold ==="
python  -m $SRC.pipeline features --split train --jobs "$JOBS"
python  -m $SRC.pipeline train --split train
python  -m $SRC.pipeline tune --split train

echo "=== [7/7] Test inference ==="
python  -m $SRC.pipeline prepare --split test --jobs "$JOBS"
python  -m $SRC.pipeline index --split test
python  -m $SRC.pipeline block --split test --jobs "$JOBS"
python  -m $SRC.pipeline features --split test --jobs "$JOBS"
python  -m $SRC.pipeline infer --split test

echo
echo "Done. Outputs under output/:"
ls -la output/
echo
echo "Optional validation (from the student_resource dir):"
echo "  python  utils/validate_submission.py --matching output/matching_results.tsv \\"
echo "        --candidate output/candidate_pairs.tsv --test-dir dataset/test"