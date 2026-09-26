"""Threshold tuning (pipeline step 8).

Sweeps the decision threshold on the validation split, optimizing the exact
competition metric (macro-averaged per-entity F0.5, precision weighted 2x)
rather than F1. Reports the best threshold with its macro precision / recall /
F0.5 and the F1 at that threshold for context.

Relies on ``artifacts/{split}/model/val_predictions.parquet`` produced by the
training step (model probability per validation pair).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import Config
from .data_io import artifact_path, ensure_dir
from .f05_scorer import f0_5


def score_at_threshold(proba: pd.Series, q: np.ndarray, y: np.ndarray, pos_arr: np.ndarray,
                       threshold: float) -> tuple[float, float, float]:
    """Macro F0.5, precision, recall for one threshold."""
    pred = (proba.to_numpy(np.float64) >= threshold).astype(np.float32)
    correct = pred * y.astype(np.float32)
    n = int(q.max()) + 1
    pred_c = np.bincount(q, weights=pred, minlength=n)
    corr_c = np.bincount(q, weights=correct, minlength=n)
    # only entities that have true matches are scored (all in train GT)
    with_pos = pos_arr > 0
    p_q = np.divide(corr_c, pred_c, out=np.zeros(n, np.float64), where=pred_c > 0)
    r_q = np.divide(corr_c, pos_arr, out=np.zeros(n, np.float64), where=pos_arr > 0)
    f_q = np.zeros(n, np.float64)
    pv, rv = p_q[with_pos], r_q[with_pos]
    denom = 0.25 * pv + rv
    with np.errstate(divide="ignore", invalid="ignore"):
        f_v = np.divide(1.25 * pv * rv, denom, out=np.zeros_like(denom), where=denom > 0)
    f_q[with_pos] = np.where((pv > 0) | (rv > 0), f_v, 0.0)
    return float(np.mean(f_q[with_pos])), float(np.mean(p_q[with_pos])), float(np.mean(r_q[with_pos]))


def run_tune(split: str, cfg: Config) -> dict:
    tcfg = cfg.tune
    df = pd.read_parquet(artifact_path(split, "model") / "val_predictions.parquet")
    q = df["q_idx"].to_numpy(np.int64)
    y = df["y"].to_numpy(np.float32)
    pos_arr = np.bincount(q, weights=y, minlength=int(q.max()) + 1)

    best = None
    thresholds = np.arange(tcfg.threshold_min, tcfg.threshold_max + 1e-9, tcfg.threshold_step)
    print(f"[tune] sweeping {len(thresholds)} thresholds on {len(df):,} val rows")
    rows = []
    for t in thresholds:
        f, p, r = score_at_threshold(df["proba"], q, y, pos_arr, float(t))
        rows.append((float(t), p, r, f))
        if best is None or f > best[3]:
            best = (float(t), p, r, f)

    # F1 for context
    f1_best = None
    for t, p, r, f in rows:
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1_best is None or f1 > f1_best[3]:
            f1_best = (t, p, r, f1)

    bt, bp, br, bf = best
    print("\n===== THRESHOLD TUNING (optimising macro F0.5) =====")
    print(f"  best threshold   : {bt:.3f}")
    print(f"  macro precision  : {bp:.4f}")
    print(f"  macro recall     : {br:.4f}")
    print(f"  macro F0.5       : {bf:.4f}")
    print(f"  best F1 at {f1_best[0]:.3f}: {f1_best[3]:.4f}")
    print(f"  (F0.5 weights precision 2x; F1 shown for comparison only)")

    out = {
        "best_threshold": float(bt),
        "best_precision": float(bp),
        "best_recall": float(br),
        "best_f0_5": float(bf),
        "n_thresholds": int(len(thresholds)),
        "sweep_summary": [(t, p, r, f) for t, p, r, f in rows],
    }
    model_dir = ensure_dir(artifact_path(split, "model"))
    with open(model_dir / "threshold.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)
    run_tune(args.split, Config())


if __name__ == "__main__":
    main()