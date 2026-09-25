# How to run the pipeline (quickstart for the person running)

## What this is

A business-entity-matching pipeline: it trains a model on the `train` split,
then predicts matches for every `source1` entity in the `test` split and writes
the submission file `output/matching_results.tsv`.

## Requirements

- Python 3.10 or newer (tested on 3.12)
- ~4GB RAM headroom, ~15GB free disk
- Runtime: roughly 2–3 hours (set a long timeout)

## Step 1 — get the code and the data

```bash
git clone <REPO_URL>
cd business_entity_resolution

# the dataset folder is NOT in the repo; you need it locally:
#   student_resource/dataset/  containing  train/  and  test/
ln -s /path/to/student_resource/dataset dataset

python3 -V   # must be 3.10+
```

Make sure `dataset/train/` and `dataset/test/` now resolve:

```bash
ls dataset/train/ dataset/test/
```

## Step 2 — install dependencies

```bash
pip install -r requirements.txt
pip install lightgbm
```

Some systems protect the system Python ("externally managed"). If `pip install`
refuses, add `--break-system-packages`:

```bash
pip install --break-system-packages lightgbm
```

## Step 3 — run the full pipeline

```bash
bash run_all.sh 16
```

The `16` is the number of parallel workers (use 8 on small machines). This
trains on `train` (prepare → blocking → features → model → threshold), then
runs inference on `test`. Equivalent step-by-step:

```bash
python3 -m src.pipeline full-train --jobs 16
python3 -m src.pipeline full-test  --jobs 16
```

## Step 4 — the file to send back

```
output/matching_results.tsv
```

(Optional) verify before sending — from the student_resource folder:

```bash
python3 utils/validate_submission.py \
  --matching <project>/output/matching_results.tsv \
  --candidate <project>/output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

It should print `PASS — no blocking issues found. Safe to submit.`

## Troubleshooting

- **`MemoryError`** during `prepare`/`block`: lower the jobs count (`bash run_all.sh 8`).
- **Data already processed (crash mid-run):** steps skip existing artifacts; just
  re-run `bash run_all.sh 16` and it resumes.
- **Pinned package conflicts:** `pip install -r requirements.txt` with a clean
  venv is recommended (`python3 -m venv .venv && source .venv/bin/activate`).
- **No `dataset` symlink:** every `prepare` run will fail with a missing file error;
  the link in Step 1 fixes it.

## Team

Power Puff Girls