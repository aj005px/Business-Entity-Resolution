"""Orchestrator: runs every pipeline step in the right order.

Usage:
    python -m src.pipeline <step> [--split ...] [--jobs N]

steps:
    eda             step 1 -- exploratory analysis (train)
    prepare         step 2 -- normalize TSVs -> parquet caches
    index           step 3a-- build TF-IDF inverted indexes
    block           step 3b-- candidate generation -> candidate chunks
    recall          step 4 -- blocking recall ceiling vs ground truth
    gt              step 6a-- build ground-truth position pairs
    features        step 5+6 -- pair features (+ labels for train)
    train           step 7 -- train model, emit validation predictions
    tune            step 8 -- sweep threshold, optimise macro F0.5
    infer           step 10 -- test inference -> output/*.tsv

    full-train      run everything needed for training in one go
    full-test       run everything needed for test inference in one go
"""
from __future__ import annotations

import argparse
import gc

from .config import Config
from . import blocking, data_io, eda, features, inference, label, prepare
from . import recall_check, threshold_tuning, train_model


def run_index(split: str, cfg: Config, fields=("name", "addr")) -> None:
    for field in fields:
        missing = [f for f in (f"{field}_index.npz", f"{field}_meta.npz",
                               f"{field}_query_tfidf.npz")
                   if not data_io.artifact_path(split, f).exists()]
        if not missing:
            print(f"[index {split}] {field} index already built, skipping")
            continue
        print(f"[index {split}] building {field} index ...")
        blocking.build_index(split, field, cfg.blocking, cfg.normalize)
        gc.collect()


def run_full_train(cfg: Config, jobs: int) -> None:
    prepare.run_prepare("train", cfg, workers=jobs)
    label.build_gt_pairs("train")
    run_index("train", cfg)
    blocking.run_block("train", cfg, jobs=jobs)
    recall_check.run_recall("train", cfg)
    features.run_features("train", cfg)
    train_model.run_train("train", cfg)
    threshold_tuning.run_tune("train", cfg)


def run_full_test(cfg: Config, jobs: int) -> None:
    prepare.run_prepare("test", cfg, workers=jobs)
    run_index("test", cfg)
    blocking.run_block("test", cfg, jobs=jobs)
    features.run_features("test", cfg)
    inference.run_inference("test", cfg)
    print("\nRun the official validator with, from student_resource/:\n"
          "  python3 utils/validate_submission.py --matching output/matching_results.tsv \\\n"
          "        --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids\n")


STEPS = {
    "eda": lambda a, c: eda.run_eda(a.split or "train", limit=a.limit),
    "prepare": lambda a, c: prepare.run_prepare(a.split, c, workers=a.jobs),
    "index": lambda a, c: run_index(a.split, c),
    "block": lambda a, c: blocking.run_block(a.split, c,
                                             start=a.start, end=a.end, jobs=a.jobs),
    "recall": lambda a, c: recall_check.run_recall(a.split, c),
    "gt": lambda a, c: label.build_gt_pairs(a.split),
    "features": lambda a, c: features.run_features(
        a.split, c, start_chunk=a.start_chunk, end_chunk=a.end_chunk),
    "train": lambda a, c: train_model.run_train(a.split, c, max_rows=a.max_rows),
    "tune": lambda a, c: threshold_tuning.run_tune(a.split, c),
    "infer": lambda a, c: inference.run_inference(a.split, c, threshold=a.threshold),
    "full-train": lambda a, c: run_full_train(c, a.jobs),
    "full-test": lambda a, c: run_full_test(c, a.jobs),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--split", default="train", help="train|test")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=20_000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--end-chunk", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--max-rows", type=int, default=80_000_000)
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="override candidate cap per source1 entity")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.max_candidates:
        cfg.blocking.max_candidates_per_query = args.max_candidates
    if args.split not in ("train", "test"):
        raise SystemExit("--split must be train or test")
    STEPS[args.step](args, cfg)


if __name__ == "__main__":
    main()