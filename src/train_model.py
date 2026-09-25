"""Model training (pipeline step 7).

Trains a gradient-boosted binary classifier on the engineered pair features.

Model choice (all open / permissive licenses):
    * LightGBM          (MIT)      -- default when installed
    * XGBoost           (Apache-2.0) -- fallback
    * sklearn GradientBoostingClassifier (BSD) -- final fallback

Validation is split *by source1_entity_id* (never by pair), matching how the
scorer aggregates predictions per entity and preventing row leakage.

Pipeline:
    1. Count positives per source1 entity (pass 0 over feature chunks).
    2. Assign each source1 entity to train / validation (seeded).
       - train:   all positives + up to ``neg_per_pos`` negatives per entity
       - val:     the full candidate space (needed for exact F0.5 tuning)
    3. Fit with early stopping on validation log-loss.
    4. Emit validation probabilities + labels for threshold tuning.

Outputs under ``artifacts/model/``: model files + ``val_predictions.parquet``.
"""
from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data_io import artifact_path, ensure_dir

# feature columns used by the model (excludes pair id + label)
FEATURE_COLS = [
    "name_lev", "addr_lev",
    "name_token_jaccard", "addr_token_jaccard",
    "name_token_overlap", "addr_token_overlap",
    "name_trigram", "addr_trigram",
    "name_len_diff", "addr_len_diff",
    "country_match",
    "name_char_len_q", "name_char_len_r",
    "addr_char_len_q", "addr_char_len_r",
    "name_tfidf_cos", "addr_tfidf_cos", "block_score",
]


def _assign_splits(n_queries: int, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = np.arange(n_queries)
    rng.shuffle(q)
    n_val = int(n_queries * val_frac)
    val = np.zeros(n_queries, bool)
    val[q[:n_val]] = True
    return val


def _pos_per_query(file_paths, n_queries: int) -> np.ndarray:
    counts = np.zeros(n_queries, np.int32)
    for f in file_paths:
        df = pd.read_parquet(f, columns=["q_idx", "label"])
        if "label" not in df.columns:
            continue
        pos = df[df["label"] == 1]
        if len(pos):
            counts += np.bincount(pos["q_idx"].to_numpy(np.int32),
                                  minlength=n_queries).astype(np.int32)
    return counts


def _training_rows(df: pd.DataFrame, val_flag: np.ndarray, allowed_neg: np.ndarray):
    """Boolean mask over a feature chunk: train positives + sampled negatives.

    df columns include q_idx, label. ``allowed_neg[q]`` = how many negatives to
    keep for entity q. Validation rows are kept whole (all negatives too).
    """
    q = df["q_idx"].to_numpy(np.int64)
    lab = df["label"].to_numpy(np.int8)
    is_val = val_flag[q]
    is_pos = lab == 1
    if not np.any(~is_val & ~is_pos):
        return is_val | is_pos
    neg = ~is_pos
    # per-entity running index among negatives (keeps file order)
    tmp = pd.DataFrame({"q": q[neg]})
    rank = tmp.groupby("q").cumcount().to_numpy()
    allowed = allowed_neg[q[neg]]
    keep_neg = rank < allowed
    neg_mask = np.zeros(len(df), bool)
    neg_idx = np.flatnonzero(neg)
    neg_mask[neg_idx[keep_neg]] = True
    return is_val | is_pos | neg_mask


def run_train(split: str, cfg: Config, max_rows: int = 80_000_000) -> dict:
    tcfg = cfg.train
    feat_dir = artifact_path(split, "features")
    files = sorted(feat_dir.glob("feat_*.parquet"))
    model_dir = ensure_dir(artifact_path(split, "model"))
    n_queries = len(pd.read_parquet(artifact_path(split, "source1.parquet"), columns=["entity_id"]))

    gc.collect()
    print(f"[train] computing positives per entity ({len(files)} files) ...")
    pos_q = _pos_per_query(files, n_queries)
    val_flag = _assign_splits(n_queries, tcfg.val_frac, tcfg.seed)
    allowed_neg = np.ceil(tcfg.subsample_neg_per_pos * pos_q).astype(np.int64)

    X_lst, y_lst, Xv_lst, yv_lst, qv_lst, rv_lst = [], [], [], [], [], []
    n_train = 0
    for c, f in enumerate(files):
        df = pd.read_parquet(f)
        mask = _training_rows(df, val_flag, allowed_neg)
        if mask.sum() == 0:
            del df
            continue
        sub = df[mask]
        is_val_row = val_flag[sub["q_idx"].to_numpy(np.int64)]
        train = sub[~is_val_row]
        val = sub[is_val_row]
        if len(train):
            X_lst.append(train[FEATURE_COLS].to_numpy(np.float32))
            y_lst.append(train["label"].to_numpy(np.int8))
            n_train += len(train)
        if len(val):
            Xv_lst.append(val[FEATURE_COLS].to_numpy(np.float32))
            yv_lst.append(val["label"].to_numpy(np.int8))
            qv_lst.append(val["q_idx"].to_numpy(np.int32))
            rv_lst.append(val["r_idx"].to_numpy(np.int32))
        del df, sub, train, val
        if c % 10 == 0:
            print(f"[train] file {c+1}/{len(files)} train_rows={n_train:,}", flush=True)
        if n_train >= max_rows:
            print(f"[train] reached row cap ({max_rows:,})")
            break
    print(f"[train] assembling ...")
    X = np.vstack(X_lst)
    y = np.concatenate(y_lst)
    Xv = np.vstack(Xv_lst) if Xv_lst else np.empty((0, len(FEATURE_COLS)), np.float32)
    yv = np.concatenate(yv_lst) if yv_lst else np.empty(0, np.int8)
    qv = np.concatenate(qv_lst) if qv_lst else np.empty(0, np.int32)
    rv = np.concatenate(rv_lst) if rv_lst else np.empty(0, np.int32)
    del X_lst, y_lst, Xv_lst, yv_lst, qv_lst, rv_lst

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 and neg > 0 else 1.0
    print(f"[train] train rows={len(y):,} pos={pos:,} neg={neg:,} "
          f"| val rows={len(yv):,} pos={(yv==1).sum():,} | scale_pos_weight={scale_pos_weight:.1f}")

    model, is_lgbm = _fit(X, y, Xv, yv, scale_pos_weight, cfg)

    # validation predictions for threshold tuning
    proba_v = model.predict_proba(Xv)[:, 1]
    pd.DataFrame({
        "q_idx": qv, "r_idx": rv, "proba": proba_v.astype(np.float32),
        "y": yv.astype(np.int8),
    }).to_parquet(model_dir / "val_predictions.parquet", index=False)

    params = {
        "model_backend": "lightgbm" if is_lgbm else _backend_name(),
        "scale_pos_weight": float(scale_pos_weight),
        "val_frac": tcfg.val_frac,
        "n_train_rows": int(len(y)),
        "n_val_rows": int(len(yv)),
        "feature_columns": FEATURE_COLS,
    }
    with open(model_dir / "params.json", "w") as fh:
        json.dump(params, fh, indent=2)

    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        for name, imp in sorted(zip(FEATURE_COLS, fi), key=lambda t: -t[1]):
            print(f"  fi {imp:6.1f}  {name}")

    # early-stop best iteration reporting
    if is_lgbm and getattr(model, "best_iteration_", None):
        print(f"[train] best iteration={model.best_iteration_}")
    del X, Xv, pos_q, val_flag, allowed_neg
    gc.collect()
    return params


def _fit(X, y, Xv, yv, scale_pos_weight, cfg: Config):
    tcfg = cfg.train
    kw = dict(learning_rate=tcfg.learning_rate, n_estimators=tcfg.n_estimators,
              random_state=tcfg.seed, n_jobs=tcfg.n_jobs)
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            num_leaves=tcfg.num_leaves, max_depth=-1,
            min_child_samples=tcfg.min_data_in_leaf,
            colsample_bytree=tcfg.feature_fraction,
            subsample=tcfg.bagging_fraction, subsample_freq=tcfg.bagging_freq,
            scale_pos_weight=scale_pos_weight, verbosity=-1, **kw,
        )
        model.fit(X, y, eval_set=[(Xv, yv)],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(tcfg.early_stopping_rounds, verbose=False)])
        model_dir = artifact_path("train", "model")
        ensure_dir(model_dir)
        model.booster_.save_model(str(model_dir / "model.txt"))
        return model, True
    except ImportError:
        pass
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            max_depth=8, learning_rate=tcfg.learning_rate, n_estimators=tcfg.n_estimators,
            tree_method="hist", scale_pos_weight=scale_pos_weight,
            subsample=tcfg.bagging_fraction, colsample_bytree=tcfg.feature_fraction,
            early_stopping_rounds=tcfg.early_stopping_rounds, random_state=tcfg.seed,
            n_jobs=tcfg.n_jobs, verbosity=0,
        )
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        model.save_model(str(artifact_path("train", "model") / "model.json"))
        return model, False
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=min(200, tcfg.n_estimators),
        learning_rate=tcfg.learning_rate, max_depth=8, subsample=0.9,
        random_state=tcfg.seed,
    )
    model.fit(X, y)
    import joblib

    model_dir = artifact_path("train", "model")
    ensure_dir(model_dir)
    joblib.dump(model, model_dir / "model_gbc.pkl")
    return model, False


def _backend_name():
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except ImportError:
        return "sklearn"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-rows", type=int, default=80_000_000)
    args = ap.parse_args(argv)
    run_train(args.split, Config(), max_rows=args.max_rows)


if __name__ == "__main__":
    main()