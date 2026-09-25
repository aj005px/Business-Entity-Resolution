"""Exploratory data analysis (pipeline step 1).

Loads the competition files, reports:
  * column presence / null counts for every source file
  * ground-truth match cardinality (0 / 1 / >1 matches per source-1 entity)
  * a random sample of raw + normalized name/address rows (to eyeball noise)

Run:  python -m src.eda --split train [--limit 20000]
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .data_io import source_path
from .normalize import normalize_address, normalize_name


def null_counts(split: str) -> dict:
    """Scan every source file fully and count empty cells per column."""
    out = {}
    for src in (1, 2, 3):
        path = source_path(split, src)
        counts = Counter()          # col -> num empty
        totals = Counter()
        for part in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                chunksize=1_000_000, low_memory=False):
            for col in part.columns:
                totals[col] += len(part)
                counts[col] += int((part[col].str.strip() == "").sum())
        out[f"source{src}"] = {c: counts[c] for c in totals}
        out[f"source{src}"]["rows"] = totals["entity_id"]
    return out


def gt_cardinality(split: str) -> dict:
    """Count how many source-1 entities have 0 / 1 / many ground-truth matches."""
    from .config import DATA_DIR

    gt_path = DATA_DIR / split / "train_ground_truth.tsv"

    def parse_count(row: str):
        parts = row.split("\t")
        if len(parts) < 2 or parts[1].strip() == "":
            return 0
        return len([x for x in parts[1].split(",") if x])

    zero = one = many = 0
    total = 0
    lengths = []
    with open(gt_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if not line.strip():
                continue
            k = parse_count(line)
            total += 1
            lengths.append(k)
            if k == 0:
                zero += 1
            elif k == 1:
                one += 1
            else:
                many += 1
    return {"total_s1": total, "0_matches": zero, "1_match": one,
            "many_matches": many, "mean_matches": float(np.mean(lengths) if lengths else 0.0)}


def noisy_samples(split: str, limit: int, n: int) -> list:
    """Return ``n`` raw rows with their normalized forms (noise eyeballing)."""
    rows = []
    s1 = pd.read_csv(source_path(split, 1), sep="\t", dtype=str, keep_default_na=False,
                     nrows=limit, low_memory=False)
    for _, r in s1.sample(min(n, len(s1)), random_state=0).iterrows():
        rows.append({
            "entity_id": r["entity_id"],
            "name": r["business_name"],
            "norm_name": normalize_name(r["business_name"]),
            "address": r["business_address"],
            "norm_addr": normalize_address(r["business_address"]),
            "country": r["country"],
        })
    return rows


def run_eda(split: str, limit: int = 20_000) -> None:
    print(f"==== EDA [{split}] ====")
    print("\n-- null (empty) counts per source column --")
    for k, v in null_counts(split).items():
        print(f"  {k}: rows={v['rows']}")
        for col, cnt in v.items():
            if col != "rows":
                print(f"     {col:20s} empty={cnt:9d}  ({100.0*cnt/v['rows']:.2f}%)")

    try:
        g = gt_cardinality(split)
        print("\n-- ground-truth match cardinality --")
        print(f"  source1 entities      : {g['total_s1']:,}")
        print(f"  0 matches             : {g['0_matches']:,}")
        print(f"  1 match               : {g['1_match']:,}")
        print(f"  >1 matches            : {g['many_matches']:,}")
        print(f"  mean matches/entity   : {g['mean_matches']:.3f}")
    except FileNotFoundError:
        print("\n  (ground truth only exists for the train split)")

    print(f"\n-- {min(limit, 6)} sample rows (raw -> normalized) --")
    for r in noisy_samples(split, limit, n=min(6, max(1, limit))):
        print(f"  {r['entity_id']}")
        print(f"    name   : {r['name']!r:<70} -> {r['norm_name']!r}")
        print(f"    address: {r['address']!r:<70} -> {r['norm_addr']!r}")
        print(f"    country: {r['country']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=20_000)
    args = ap.parse_args(argv)
    run_eda(args.split, limit=args.limit)


if __name__ == "__main__":
    main()