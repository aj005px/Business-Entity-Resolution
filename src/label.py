"""Ground-truth pairs and labeling (pipeline step 6).

Converts ``train_ground_truth.tsv`` (source-1 id -> comma-separated matched ids)
into position-keyed positive pairs ``(q_pos, r_pos)`` so they can be joined
against candidate pairs cheaply and vectorised.

Persists:
    artifacts/{split}/gt_pairs.parquet   positives only (q_pos, r_pos)

Note: rather than materialising a giant merged ``pairs.parquet``, labeling is
done *streaming per candidate chunk* inside ``features.py``, which is what the
training stage consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data_io import artifact_path, load_df


def build_id_position_maps(split: str):
    """Return (query_id_to_pos, ref_id_to_pos) dicts for a split."""
    q = load_df(artifact_path(split, "source1.parquet"))
    r = load_df(artifact_path(split, "refs.parquet"))
    qmap = {eid: i for i, eid in enumerate(q["entity_id"].tolist())}
    rmap = {eid: i for i, eid in enumerate(r["entity_id"].tolist())}
    del q, r
    return qmap, rmap


def build_gt_pairs(split: str) -> None:
    """Stream ground truth and persist position-keyed positive pairs."""
    out = artifact_path(split, "gt_pairs.parquet")
    if out.exists():
        print(f"[label] {out} already exists")
        return
    qmap, rmap = build_id_position_maps(split)
    gt_path = DATA_DIR / split / "train_ground_truth.tsv"
    qs, rs = [], []
    with open(gt_path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qpos = qmap.get(parts[0])
            if qpos is None:
                continue
            for cid in parts[1].split(","):
                if not cid:
                    continue
                rpos = rmap.get(cid)
                if rpos is not None:
                    qs.append(qpos)
                    rs.append(rpos)
    df = pd.DataFrame({"q_pos": np.asarray(qs, np.int32), "r_pos": np.asarray(rs, np.int32)})
    df.to_parquet(out, index=False)
    print(f"[label] gt_pairs: {len(df):,} positive pairs")
    del qmap, rmap, df


def load_label_keys(split: str) -> np.ndarray:
    """Sorted int64 keys ``(q<<32)|r`` of all true positive pairs."""
    df = load_df(artifact_path(split, "gt_pairs.parquet"))
    keys = (df["q_pos"].to_numpy(np.int64) << 32) | df["r_pos"].to_numpy(np.int64)
    return np.sort(keys)


def label_pairs(q_pos: np.ndarray, r_pos: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Vectorised 1/0 membership of pairs against ground-truth positive keys."""
    pair_keys = (q_pos.astype(np.int64) << 32) | r_pos.astype(np.int64)
    idx = np.searchsorted(keys, pair_keys)
    idx = np.clip(idx, 0, len(keys) - 1)
    return (keys[idx] == pair_keys).astype(np.int8)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    build_gt_pairs(args.split)