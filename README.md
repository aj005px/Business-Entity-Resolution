# Business Entity Resolution — Power Puff Girls

Match noisy `source2`/`source3` business records to `source1` entities using
only `business_name`, `business_address` and `country`. The competition metric
is the **macro-averaged per-source1-entity F0.5** (precision weighted 2x).

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --break-system-packages lightgbm   # on PEP-668-locked systems
ln -s /path/to/6ab10eb3b23ba_student_resource/student_resource/dataset ./dataset
```

Raw TSVs must live at `dataset/<split>/<split>_source{1,2,3}.tsv` and
`dataset/train/train_ground_truth.tsv`. That path and the artifact/output
directories can be redirected with `ER_DATA_DIR` / `ER_ARTIFACT_DIR`.

## Pipeline

Everything is a single entry point:

```bash
python3 -m src.pipeline <step> [--split train|test] [--jobs N] [--max-candidates M]
```

Run the steps **in this order** (or use the `full-train` / `full-test` shorthands
which do it all in sequence):

| # | step        | split        | what it does |
|---|-------------|--------------|--------------|
| 1 | `prepare`   | train, test  | normalize name/address (NFKD + diacritic stripping, tokenization), country int encoding → `artifacts/{split}/source{1,2,3}.parquet` + `refs.parquet` |
| 2 | `gt`        | train        | build positive (query, ref) pairs from `train_ground_truth.tsv` |
| 3 | `index`     | train, test  | per-field (name/addr) TF-IDF vocab + inverted-index postings + query vectors |
| 4 | `block`     | train, test  | candidate generation: country hard-filter, rare-token blocking, exact TF-IDF cosine top-k, score floor + cap → `candidates/cand_*.parquet` |
| 5 | `recall`    | train        | blocker recall ceiling vs ground truth |
| 6 | `features`  | train, test  | per-pair similarity features (Levenshtein, token/trigram Dice, cosine, length gaps) → `feat_*.parquet` |
| 7 | `train`     | train        | LightGBM ranking classifier on labeled pairs (per-entity negative sampling) → `model/model.txt` + `val_predictions.parquet` |
| 8 | `tune`      | train        | sweep decision threshold optimizing macro F0.5 → `model/threshold.json` |
| 9 | `infer`     | test         | score test pairs, group by entity, write `output/` files |

### Full runs

```bash
python3 -m src.pipeline full-train --jobs 16     # evaluates every step on train
python3 -m src.pipeline full-test  --jobs 16     # produces the submission files
```

`block` is parallel (pass `--jobs N`); results are byte-identical to the
single-process run. Each chunk of queries is written independently so reruns
only recompute missing chunks.

## Output

`infer` writes:

- `output/candidate_pairs.tsv` — every `source1_entity_id \t matched_entity_ids`
  line, one per test entity, populated with candidate pairs.
- `output/matching_results.tsv` — the actual prediction (empty `matched_entity_ids`
  when nothing passes threshold).

Validate with the official checker (run from `6ab10eb3b23ba_student_resource/student_resource/`):

```bash
python3 utils/validate_submission.py \
  --matching <repo>/output/matching_results.tsv \
  --candidate <repo>/output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

## Design decisions

- **Country blocking is free recall.** Empirical check on train ground truth:
  100% of true matches share country, so it is a hard filter.
- **Candidate ranking policy** (`_apply_floor_cap` in `src/blocking.py`):
  1. keep every ref with best-field cosine `>= nn_similarity_threshold`;
  2. always keep the per-field top-k cosine refs (rescues weak-but-best-field
     matches, e.g. businesses sharing only an address);
  3. guarantee `>= min_candidates_per_query`;
  4. cap the merged pool at `max_candidates_per_query` ranked by
     `max(name_cos, addr_cos)`.
  This bounds generic-name queries to ~75 candidates/query on the dev slice
  while keeping a ~96.7% recall ceiling.
- **Two fields, two vocabularies.** Name and address get separate IDFs; the
  alias cases ("Solkeloquo" ↔ "Uptown Pub", same address) are recovered by the
  address block.
- **Training labels.** Positives come from ground truth; each source1 entity
  also gets `subsample_neg_per_pos` negatives sampled from its candidate set so
  the classifier sees the same distribution at inference time. Validation
  entities are disjoint from train entities.
- **Threshold on the true metric.** The decision threshold is chosen by sweeping
  macro F0.5 directly (not F1).

## Module map (`src/`)

|  file | role |
|------|------|
| `config.py`     | paths + per-step settings |
| `data_io.py`    | TSV/parquet I/O |
| `normalize.py`  | normalization, tokenization, char-trigram Dice |
| `prepare.py`    | normalize raw TSVs → parquet caches |
| `blocking.py`   | index build + candidate generation (postings TF-IDF NN) |
| `features.py`   | per-pair feature computation |
| `label.py`      | ground-truth pair construction / labeling |
| `f05_scorer.py` | the exact competition metric |
| `recall_check.py` | blocker recall ceiling report |
| `train_model.py` | classifier training |
| `threshold_tuning.py` | F0.5 threshold sweep |
| `inference.py`  | predictions + submission TSVs |
| `pipeline.py`   | step orchestration |

## Reference (dev-slice numbers, 30k entities)

- Blocker: ~75 candidates/query, recall ceiling 96.7%, entity-level capture 99.6%.
- Validation model: macro F0.5 **0.9766** @ threshold 0.920 (P 0.986 / R 0.959).
- Official validator: PASS on the dev-slice submission.