"""Inference (pipeline step 10).

Runs the trained model over the test split's candidate pairs, applies the tuned
threshold, and writes the two required submission files to ``output/``:

    output/candidate_pairs.tsv     source1_entity_id, candidate_entity_ids
    output/matching_results.tsv    source1_entity_id, matched_entity_ids

Guarantees:
  * every test source-1 entity appears exactly once (empty = no match)
  * no duplicate ids inside any list
  * matched ids are only S2-/S3- prefixed ids that exist in the test set
  * final matches are a strict subset of the blocking candidates
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, OUTPUT_DIR
from .data_io import artifact_path, ensure_dir, load_df
from .train_model import FEATURE_COLS

REFS_TABLE = "refs.parquet"
Q_TABLE = "source1.parquet"


def _load_model(train_split: str, backend: str):
    model_dir = artifact_path(train_split, "model")
    if backend == "lightgbm":
        import lightgbm as lgb

        bst = lgb.Booster(model_file=str(model_dir / "model.txt"))
        return lambda X: bst.predict(X)
    if backend == "xgboost":
        import xgboost as xgb

        bst = xgb.Booster()
        bst.load_model(str(model_dir / "model.json"))
        try:
            end = int(bst.attr("best_iteration"))
            n_trees = bst.num_boosted_rounds()
            if 0 <= end < n_trees:
                return lambda X: bst.predict(xgb.DMatrix(X), iteration_range=(0, end + 1))
            print(f"[infer] best_iteration={end} >= n_trees={n_trees}; using all trees")
        except (AttributeError, TypeError, ValueError):
            pass
        return lambda X: bst.predict(xgb.DMatrix(X))
    import joblib

    model = joblib.load(model_dir / "model_gbc.pkl")
    return lambda X: model.predict_proba(X)[:, 1]


def _detect_backend(train_split: str) -> str:
    model_dir = artifact_path(train_split, "model")
    if (model_dir / "model.txt").exists():
        return "lightgbm"
    if (model_dir / "model.json").exists():
        return "xgboost"
    if (model_dir / "model_gbc.pkl").exists():
        return "sklearn"
    raise FileNotFoundError(f"no trained model under {model_dir} -- run pipeline train first")


def run_inference(test_split: str, cfg: Config, threshold: float | None = None) -> dict:
    train_split = "train"
    backend = _detect_backend(train_split)
    model_dir = artifact_path(train_split, "model")
    if threshold is None:
        threshold = json.load(open(model_dir / "threshold.json"))["best_threshold"]
    predict = _load_model(train_split, backend)

    feat_files = sorted(artifact_path(test_split, "features").glob("feat_*.parquet"))
    total_rows = 0
    pred_parts = []
    for f in feat_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        X = df[FEATURE_COLS].to_numpy(np.float32)
        proba = predict(X)
        proba = np.asarray(proba, np.float32).ravel()
        pred_parts.append(pd.DataFrame({
            "q_idx": df["q_idx"].to_numpy(np.int32),
            "r_idx": df["r_idx"].to_numpy(np.int32),
            "proba": proba,
        }))
        total_rows += len(df)
        if len(pred_parts) % 10 == 0:
            print(f"[infer] predicted {total_rows:,} pairs", flush=True)
        del df, X
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame(
        columns=["q_idx", "r_idx", "proba"])
    del pred_parts
    print(f"[infer] total candidate pairs with scores: {len(pred):,}")

    # id lookup tables
    queries = load_df(artifact_path(test_split, Q_TABLE))
    refs = load_df(artifact_path(test_split, REFS_TABLE))
    q_ids = queries["entity_id"].to_numpy()
    r_ids = refs["entity_id"].to_numpy()
    n_queries = len(queries)

    ensure_dir(OUTPUT_DIR)
    cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
    match_path = OUTPUT_DIR / "matching_results.tsv"
    BLOCK = 50_000
    cand_w = open(cand_path, "w", encoding="utf-8", newline="\n")
    match_w = open(match_path, "w", encoding="utf-8", newline="\n")
    cand_w.write("source1_entity_id\tcandidate_entity_ids\n")
    match_w.write("source1_entity_id\tmatched_entity_ids\n")

    matched_total = 0
    n_with_match = 0
    for q0 in range(0, n_queries, BLOCK):
        q1 = min(q0 + BLOCK, n_queries)
        rows = pred[(pred["q_idx"] >= q0) & (pred["q_idx"] < q1)] if len(pred) else pred
        if len(rows):
            rows = rows.sort_values(["q_idx", "proba"], ascending=[True, False])
        grouped = rows.groupby("q_idx") if len(rows) else ()
        gdict = {k: g for k, g in grouped}
        for q in range(q0, q1):
            qid = q_ids[q]
            g = gdict.get(q)
            cand_ids = []
            match_ids = []
            if g is not None and len(g):
                cand_list = g["r_idx"].to_numpy(np.int64)
                cand_ids = _dedupe_ids(r_ids[cand_list])
                m = g[g["proba"].to_numpy(np.float32) >= threshold]["r_idx"].to_numpy(np.int64)
                match_ids = _dedupe_ids(r_ids[m])
                if len(match_ids):
                    matched_total += len(match_ids)
                    n_with_match += 1
            cand_w.write(f"{qid}\t" + (",".join(cand_ids) if cand_ids else "") + "\n")
            match_w.write(f"{qid}\t" + (",".join(match_ids) if match_ids else "") + "\n")
        del grouped, gdict, rows
        if q1 % (BLOCK * 5) == 0 or q1 == n_queries:
            print(f"[infer] wrote {q1:,}/{n_queries:,} entities "
                  f"(matched={matched_total:,})", flush=True)
    cand_w.close()
    match_w.close()

    print("\n===== INFERENCE SUMMARY =====")
    print(f"  test entities        : {n_queries:,}")
    print(f"  candidate pairs      : {len(pred):,}")
    print(f"  matched ids          : {matched_total:,}")
    print(f"  entities w/ any match: {n_with_match:,}")
    print(f"  threshold            : {threshold:.3f}")
    print(f"  wrote {cand_path}")
    print(f"  wrote {match_path}")
    return {"n_entities": n_queries, "candidate_pairs": len(pred),
            "matched_ids": int(matched_total), "entities_with_match": int(n_with_match),
            "threshold": float(threshold)}


def _dedupe_ids(id_list: np.ndarray) -> list[str]:
    """Stable-ish ordering by first appearance, drop duplicates."""
    if len(id_list) == 0:
        return []
    if len(id_list) == 1:
        return [str(id_list[0])]
    seen = set()
    out = []
    for e in id_list:
        if e not in seen:
            seen.add(e)
            out.append(str(e))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--validate", action="store_true", help="run the official validator after writing")
    args = ap.parse_args(argv)
    run_inference(args.split, Config(), threshold=args.threshold)
    if args.validate:
        script = Path(Path(__file__).resolve().parent.parent.parent) / "utils" / "validate_submission.py"
        cmd = [sys.executable, str(script), "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
               "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
               "--test-dir", str(Config().DATA_DIR / "test")]
        print(f"[infer] running: {' '.join(cmd)}")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()