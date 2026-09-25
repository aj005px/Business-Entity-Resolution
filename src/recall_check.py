"""Blocking recall check (pipeline step 4).

Computes the *recall ceiling*: of all ground-truth (source1, matched) pairs,
what fraction survive the blocking step. This is the maximum score the full
pipeline can ever achieve, no matter how good the classifier is.

Also reports candidate-set statistics (avg candidates per source1 entity,
fraction of entities with zero candidates) so blocking quality is visible.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config
from .data_io import artifact_path, load_df


def candidate_sets_for_range(chunk_frame: pd.DataFrame, start: int, end: int):
    """Build {q_pos -> set(r_pos)} from one candidate chunk."""
    sets = {}
    frame = chunk_frame[(chunk_frame["q_idx"] >= start) & (chunk_frame["q_idx"] < end)]
    for q, g in frame.groupby("q_idx"):
        sets[int(q)] = set(g["r_idx"].tolist())
    return sets


def run_recall(split: str, cfg: Config) -> dict:
    gt = load_df(artifact_path(split, "gt_pairs.parquet"))
    gt = gt.sort_values("q_pos").reset_index(drop=True)

    cand_dir = artifact_path(split, "candidates")
    files = sorted(cand_dir.glob("cand_*.parquet"))
    total = captured = 0
    q_with_matches = set(gt["q_pos"].unique().tolist())
    q_captured_any = set()

    n_pairs = 0
    n_queries_have = 0
    per_found = []
    range_l = 0
    for c, f in enumerate(files):
        cand = pd.read_parquet(f)
        if len(cand) == 0:
            continue
        lo, hi = int(cand["q_idx"].min()), int(cand["q_idx"].max()) + 1
        n_pairs += len(cand)
        n_queries_have += int(cand["q_idx"].nunique())
        per_found.append(len(cand) / max(1, int(cand["q_idx"].nunique())))
        sets = candidate_sets_for_range(cand, lo, hi)
        gts = gt[(gt["q_pos"] >= lo) & (gt["q_pos"] < hi)]
        for _, row in gts.iterrows():
            total += 1
            if row["r_pos"] in sets.get(int(row["q_pos"]), set()):
                captured += 1
                q_captured_any.add(int(row["q_pos"]))
        if c % 20 == 0:
            print(f"[recall {split}] chunk {c}/{len(files)} positives checked={total:,} captured={captured:,}", flush=True)
        del cand, sets, gts

    recall = captured / total if total else 0.0
    any_captured = len(q_captured_any & q_with_matches)
    stats = {
        "positive_pairs_in_gt": total,
        "captured_positive_pairs": captured,
        "recall_ceiling": recall,
        "entities_with_true_matches": len(q_with_matches),
        "entities_with_any_captured": any_captured,
        "entity_level_capture_rate": any_captured / max(1, len(q_with_matches)),
        "candidate_pairs_total": n_pairs,
        "queries_with_candidates": n_queries_have,
        "avg_candidates_per_query": float(np.mean(per_found)) if per_found else 0.0,
    }
    print("\n===== BLOCKING RECALL CEILING =====")
    for k, v in stats.items():
        if "recall" in k or "rate" in k:
            print(f"  {k:<32s}: {v * 100:.2f}%")
        else:
            print(f"  {k:<32s}: {v:,}")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)
    run_recall(args.split, Config())


if __name__ == "__main__":
    main()