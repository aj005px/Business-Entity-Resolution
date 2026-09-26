"""Model training (pipeline step 7).

Trains a gradient-boosted binary classifier on the engineered pair features.

Model choice (all open / permissive licenses):
    * XGBoost           (Apache-2.0) -- GPU (CUDA) capable, preferred when use_gpu
    * LightGBM          (MIT)        -- CPU baseline / fallback
    * sklearn GradientBoostingClassifier (BSD) -- only if allow_sklearn_fallback

A backend failure is reported loudly: with model_type="auto" the next backend is
tried, and if none work the step raises instead of silently degrading to a
different model family.

Validation is split *by source1_entity_id* (never by pair), matching how the
scorer aggregates predictions per entity and preventing row leakage.

Pipeline:
    1. Scan pass (2 columns)  -> positives per entity + exact train-row count.
    2. Fill pass              -> pre-allocated train matrix (no vstack doubling).
    3. Capped eval matrix     -> early-stopping signal only.
    4. Fit with early stopping on validation log-loss.
    5. Stream the *full* validation candidate space in a second pass, scoring
       shard by shard straight into ``val_predictions.parquet`` so the exact
       candidate distribution is available to the F0.5 threshold sweep without
       ever holding it in RAM.

Outputs under ``artifacts/{split}/model/``: model file + params.json +
val_predictions.parquet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
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


def _within_group_rank(q: np.ndarray) -> np.ndarray:
    """0-based rank of every element inside its equal-value group.

    Equivalent to ``pandas.groupby(q).cumcount()`` (stable, file order kept) but
    pure numpy, which matters because this runs on ~150M rows.
    """
    n = len(q)
    if n == 0:
        return np.empty(0, np.int64)
    order = np.argsort(q, kind="stable")
    qs = q[order]
    idx = np.arange(n, dtype=np.int64)
    is_new = np.empty(n, bool)
    is_new[0] = True
    np.not_equal(qs[1:], qs[:-1], out=is_new[1:])
    start = np.maximum.accumulate(np.where(is_new, idx, 0))
    rank = np.empty(n, np.int64)
    rank[order] = idx - start
    return rank


def _select_rows(q: np.ndarray, lab: np.ndarray, take_entity: np.ndarray,
                 allowed_neg: np.ndarray) -> np.ndarray:
    """Boolean mask over one feature chunk.

    Keeps rows whose entity passes ``take_entity``: all positives, plus the
    first ``allowed_neg[entity]`` negatives per entity in file order. Rows of
    other entities are always dropped, so the train and validation halves stay
    entity-disjoint.
    """
    pos = take_entity[q] & (lab == 1)
    neg_sel = take_entity[q] & (lab != 1)
    if not neg_sel.any():
        return pos
    nq = q[neg_sel]
    keep = _within_group_rank(nq) < allowed_neg[nq]
    out = pos.copy()
    out[np.flatnonzero(neg_sel)[keep]] = True
    return out


def _scan_features(files, n_queries: int, subsample: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 1 over the feature chunks: positives / negatives per entity.

    Reads only ``q_idx``/``label`` so the pass stays cheap even for ~150M rows.
    With both per-entity counts in hand the exact train-row count follows
    without a second scan:

        rows(entity) = positives + min(allowed_neg, negatives)
    """
    pos_q = np.zeros(n_queries, np.int32)
    neg_q = np.zeros(n_queries, np.int32)
    for f in files:
        df = pd.read_parquet(f, columns=["q_idx", "label"])
        if "label" not in df.columns:
            continue
        q = df["q_idx"].to_numpy(np.int64)
        lab = df["label"].to_numpy(np.int8)
        pos_q += np.bincount(q[lab == 1], minlength=n_queries).astype(np.int32)
        neg_q += np.bincount(q[lab != 1], minlength=n_queries).astype(np.int32)
    allowed_neg = np.ceil(subsample * pos_q).astype(np.int64)
    return pos_q, neg_q, allowed_neg


def _fill_train_matrix(files, X: np.ndarray, y: np.ndarray, train_entity: np.ndarray,
                       allowed_neg: np.ndarray) -> int:
    """Pass 2: write the training rows into the pre-allocated arrays."""
    filled = 0
    cap = len(y)
    for c, f in enumerate(files):
        if filled >= cap:
            break
        df = pd.read_parquet(f)
        if "label" not in df.columns:
            continue
        q = df["q_idx"].to_numpy(np.int64)
        lab = df["label"].to_numpy(np.int8)
        m = _select_rows(q, lab, train_entity, allowed_neg)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        if filled + len(idx) > cap:
            idx = idx[: cap - filled]
        n = len(idx)
        X[filled:filled + n] = df[FEATURE_COLS].to_numpy(np.float32)[idx]
        y[filled:filled + n] = lab[idx]
        filled += n
        if c % 10 == 0:
            print(f"[train] feature chunk {c+1}/{len(files)} train_rows={filled:,}", flush=True)
    return filled


def _build_eval_matrix(files, val_entity: np.ndarray, allowed_eval: np.ndarray,
                       cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Capped early-stopping set: val positives + ``val_neg_per_pos`` negatives."""
    Xs, ys = [], []
    total = 0
    for f in files:
        if cap and total >= cap:
            break
        df = pd.read_parquet(f)
        if "label" not in df.columns:
            continue
        q = df["q_idx"].to_numpy(np.int64)
        lab = df["label"].to_numpy(np.int8)
        m = _select_rows(q, lab, val_entity, allowed_eval)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        if cap and total + len(idx) > cap:
            idx = idx[: cap - total]
        n = len(idx)
        Xs.append(df[FEATURE_COLS].to_numpy(np.float32)[idx])
        ys.append(lab[idx])
        total += n
    if not Xs:
        return (np.empty((0, len(FEATURE_COLS)), np.float32), np.empty(0, np.int8))
    return np.concatenate(Xs), np.concatenate(ys)


def _predict_val_streaming(predict, files, val_entity: np.ndarray,
                           model_dir: Path) -> int:
    """Score the full validation candidate space, shard by shard.

    The prediction frame is written in row-group-sized shards and then
    concatenated with a streaming Parquet writer, so peak RAM is one shard.
    """
    shard_dir = ensure_dir(model_dir / "val_shards")
    for old in shard_dir.glob("val_*.parquet"):
        old.unlink()
    n_rows = 0
    n_shards = 0
    for c, f in enumerate(files):
        df = pd.read_parquet(f)
        m = val_entity[df["q_idx"].to_numpy(np.int64)]
        if not m.any():
            continue
        sub = df[m]
        proba = np.asarray(predict(sub[FEATURE_COLS].to_numpy(np.float32)), np.float32).ravel()
        pd.DataFrame({
            "q_idx": sub["q_idx"].to_numpy(np.int32),
            "r_idx": sub["r_idx"].to_numpy(np.int32),
            "proba": proba,
            "y": sub["label"].to_numpy(np.int8),
        }).to_parquet(shard_dir / f"val_{n_shards:05d}.parquet", index=False)
        n_rows += len(sub)
        n_shards += 1
        if c % 10 == 0:
            print(f"[train] scored val chunk {c+1}/{len(files)} rows={n_rows:,}", flush=True)
        del df, sub, proba

    import pyarrow.parquet as pq

    out = model_dir / "val_predictions.parquet"
    writer = None
    try:
        for shard in sorted(shard_dir.glob("val_*.parquet")):
            table = pq.read_table(shard)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
            del table
    finally:
        if writer is not None:
            writer.close()
        shutil.rmtree(shard_dir, ignore_errors=True)
    return n_rows


# ---------------------------------------------------------------------------
# Backends: each returns (predict_proba_fn, info_dict) and raises on failure
# ---------------------------------------------------------------------------


def _fit_xgboost(X, y, Xv, yv, scale_pos_weight, tcfg, model_dir):
    import xgboost as xgb

    device = tcfg.device if tcfg.use_gpu else "cpu"
    cuda_built = "USE_CUDA" in xgb.build_info()
    if tcfg.use_gpu and not cuda_built:
        print("[train] WARNING: use_gpu=True but this xgboost build has no CUDA "
              "support -- falling back to CPU", flush=True)
    gpu = bool(tcfg.use_gpu and cuda_built and str(device).startswith("cuda"))
    print(f"[train] fitting XGBoost  device={device} (gpu_active={gpu})  "
          f"tree_method=hist  n_estimators={tcfg.n_estimators}  rows={len(y):,}",
          flush=True)
    t0 = time.time()
    model = xgb.XGBClassifier(
        max_depth=tcfg.xgb_max_depth,
        learning_rate=tcfg.learning_rate,
        n_estimators=tcfg.n_estimators,
        tree_method="hist",
        device=device,
        max_bin=tcfg.xgb_max_bin,
        scale_pos_weight=scale_pos_weight,
        subsample=tcfg.bagging_fraction,
        colsample_bytree=tcfg.feature_fraction,
        eval_metric="logloss",
        early_stopping_rounds=tcfg.early_stopping_rounds,
        random_state=tcfg.seed,
        n_jobs=tcfg.n_jobs,
        verbosity=0,
    )
    model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    model.save_model(str(model_dir / "model.json"))
    best = getattr(model, "best_iteration", None)
    info = {
        "model_backend": "xgboost",
        "device": device,
        "gpu": gpu,
        "max_depth": tcfg.xgb_max_depth,
        "max_bin": tcfg.xgb_max_bin,
        "best_iteration": None if best is None else int(best),
        "trees_used": None if best is None else int(best) + 1,
        "fit_seconds": round(time.time() - t0, 1),
        "feature_importance_type": "gain",
        "feature_importances": {
            n: round(float(v), 6)
            for n, v in sorted(zip(FEATURE_COLS, model.feature_importances_),
                               key=lambda t: -t[1])
        },
    }
    return (lambda A: model.predict_proba(A)[:, 1]), info


def _fit_lightgbm(X, y, Xv, yv, scale_pos_weight, tcfg, model_dir):
    import lightgbm as lgb

    device = tcfg.lgbm_device if tcfg.use_gpu else "cpu"
    if device == "cuda":
        raise ValueError(
            "LightGBM's CUDA learner is not available in the Windows wheels "
            "(Linux-only); use lgbm_device='cpu' or 'gpu' (OpenCL)."
        )
    print(f"[train] fitting LightGBM  device={device}  num_leaves={tcfg.num_leaves}  "
          f"n_estimators={tcfg.n_estimators}  rows={len(y):,}", flush=True)
    t0 = time.time()
    model = lgb.LGBMClassifier(
        num_leaves=tcfg.num_leaves,
        max_depth=tcfg.max_depth,
        min_child_samples=tcfg.min_data_in_leaf,
        colsample_bytree=tcfg.feature_fraction,
        subsample=tcfg.bagging_fraction,
        subsample_freq=tcfg.bagging_freq,
        scale_pos_weight=scale_pos_weight,
        learning_rate=tcfg.learning_rate,
        n_estimators=tcfg.n_estimators,
        random_state=tcfg.seed,
        n_jobs=tcfg.n_jobs,
        device=device,
        verbosity=-1,
    )
    model.fit(X, y, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(tcfg.early_stopping_rounds, verbose=False)])
    model.booster_.save_model(str(model_dir / "model.txt"))
    info = {
        "model_backend": "lightgbm",
        "device": device,
        "gpu": device != "cpu",
        "num_leaves": tcfg.num_leaves,
        "best_iteration": int(model.best_iteration_),
        "trees_used": int(model.best_iteration_),
        "fit_seconds": round(time.time() - t0, 1),
        "feature_importance_type": "split",
        "feature_importances": {
            n: int(v) for n, v in sorted(
                zip(FEATURE_COLS, model.booster_.feature_importance("split")),
                key=lambda t: -t[1])
        },
    }
    return (lambda A: model.predict_proba(A)[:, 1]), info


def _fit_sklearn(X, y, Xv, yv, scale_pos_weight, tcfg, model_dir):
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier

    print(f"[train] fitting sklearn GradientBoostingClassifier (last resort)  "
          f"rows={len(y):,}", flush=True)
    t0 = time.time()
    model = GradientBoostingClassifier(
        n_estimators=min(200, tcfg.n_estimators),
        learning_rate=tcfg.learning_rate, max_depth=8, subsample=0.9,
        random_state=tcfg.seed,
    )
    model.fit(X, y)
    joblib.dump(model, model_dir / "model_gbc.pkl")
    return (lambda A: model.predict_proba(A)[:, 1]), {
        "model_backend": "sklearn",
        "device": "cpu",
        "gpu": False,
        "fit_seconds": round(time.time() - t0, 1),
    }


def _backend_order(tcfg) -> list[str]:
    if tcfg.model_type != "auto":
        return [tcfg.model_type]
    return ["xgboost", "lightgbm"] if tcfg.use_gpu else ["lightgbm", "xgboost"]


def _fit(X, y, Xv, yv, scale_pos_weight, cfg: Config, model_dir: Path):
    """Try each candidate backend in order; raise if none succeed."""
    tcfg = cfg.train
    errors = []
    for backend in _backend_order(tcfg):
        fn = {"xgboost": _fit_xgboost, "lightgbm": _fit_lightgbm,
              "sklearn": _fit_sklearn}.get(backend)
        if fn is None:
            raise ValueError(f"unknown model_type '{backend}'")
        try:
            return fn(X, y, Xv, yv, scale_pos_weight, tcfg, model_dir)
        except Exception as exc:  # noqa: BLE001 - reported, then next backend
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(f"{backend} -> {msg}")
            print(f"[train] backend '{backend}' FAILED ({msg})", flush=True)
    if tcfg.allow_sklearn_fallback and "sklearn" not in _backend_order(tcfg):
        print("[train] falling back to sklearn GradientBoostingClassifier", flush=True)
        return _fit_sklearn(X, y, Xv, yv, scale_pos_weight, tcfg, model_dir)
    raise RuntimeError("no usable training backend:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------


def run_train(split: str, cfg: Config, max_rows: int = 80_000_000,
              max_eval_rows: int | None = None) -> dict:
    tcfg = cfg.train
    feat_dir = artifact_path(split, "features")
    files = sorted(feat_dir.glob("feat_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no feature chunks under {feat_dir} -- run 'features' first")
    model_dir = ensure_dir(artifact_path(split, "model"))
    n_queries = len(pd.read_parquet(artifact_path(split, "source1.parquet"),
                                    columns=["entity_id"]))
    cap = max_eval_rows if max_eval_rows is not None else tcfg.max_eval_rows
    t_start = time.time()

    print(f"[train] {len(files)} feature chunks, {n_queries:,} source1 entities")
    print("[train] pass 1/3: scanning q_idx/label ...", flush=True)
    t0 = time.time()
    val_flag = _assign_splits(n_queries, tcfg.val_frac, tcfg.seed)
    train_entity = ~val_flag
    pos_q, neg_q, allowed_neg = _scan_features(files, n_queries, tcfg.subsample_neg_per_pos)
    n_train_rows = int((np.minimum(allowed_neg, neg_q)[train_entity]).sum()
                       + pos_q[train_entity].sum())
    print(f"[train] scan done in {time.time()-t0:.0f}s | positives={int(pos_q.sum()):,} "
          f"| negatives={int(neg_q.sum()):,} | train rows={n_train_rows:,}", flush=True)

    n_use = min(n_train_rows, max_rows) if max_rows else n_train_rows
    if n_use < n_train_rows:
        print(f"[train] row cap active: using {n_use:,} of {n_train_rows:,}")
    X = np.empty((n_use, len(FEATURE_COLS)), np.float32)
    y = np.empty(n_use, np.int8)
    print(f"[train] pass 2/3: filling train matrix ({n_use:,} x {len(FEATURE_COLS)}) ...",
          flush=True)
    t0 = time.time()
    filled = _fill_train_matrix(files, X, y, train_entity, allowed_neg)
    X, y = X[:filled], y[:filled]
    print(f"[train] train matrix filled in {time.time()-t0:.0f}s "
          f"({X.nbytes/2**30:.2f} GiB)", flush=True)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 and neg > 0 else 1.0
    print(f"[train] train rows={len(y):,} pos={pos:,} neg={neg:,} "
          f"| scale_pos_weight={scale_pos_weight:.1f}")

    # Early-stopping set: val positives + val_neg_per_pos negatives per entity,
    # downscaled if the natural size would blow the cap.
    allowed_eval = np.ceil(tcfg.val_neg_per_pos * pos_q).astype(np.int64)
    est_eval = int((np.minimum(allowed_eval, neg_q)[val_flag]).sum()
                   + pos_q[val_flag].sum())
    if cap and est_eval > cap:
        shrink = cap / est_eval
        allowed_eval = np.floor(allowed_eval * shrink).astype(np.int64)
        print(f"[train] eval cap active: ~{est_eval:,} -> <= {cap:,} rows "
              f"(val_neg_per_pos {tcfg.val_neg_per_pos} -> "
              f"{tcfg.val_neg_per_pos*shrink:.2f})")
    t0 = time.time()
    Xv, yv = _build_eval_matrix(files, val_flag, allowed_eval, cap)
    n_eval_rows = int(len(yv))
    print(f"[train] eval set rows={n_eval_rows:,} pos={int((yv==1).sum()):,} "
          f"in {time.time()-t0:.0f}s", flush=True)
    if n_eval_rows == 0:
        raise RuntimeError("empty evaluation set -- check the validation split")

    predict, info = _fit(X, y, Xv, yv, scale_pos_weight, cfg, model_dir)
    del Xv, yv

    print(f"[train] pass 3/3: scoring the full validation candidate space ...", flush=True)
    t0 = time.time()
    n_val_rows = _predict_val_streaming(predict, files, val_flag, model_dir)
    print(f"[train] wrote {n_val_rows:,} validation predictions in {time.time()-t0:.0f}s")

    params = {
        **info,
        "scale_pos_weight": float(scale_pos_weight),
        "val_frac": tcfg.val_frac,
        "n_train_rows": int(len(y)),
        "n_train_pos": pos,
        "n_eval_rows": n_eval_rows,
        "n_val_pred_rows": int(n_val_rows),
        "val_neg_per_pos": tcfg.val_neg_per_pos,
        "max_eval_rows": int(cap) if cap else 0,
        "subsample_neg_per_pos": tcfg.subsample_neg_per_pos,
        "feature_columns": FEATURE_COLS,
        "total_seconds": round(time.time() - t_start, 1),
    }
    with open(model_dir / "params.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(params, fh, indent=2)

    fi_type = info.get("feature_importance_type", "split")
    for name, imp in (info.get("feature_importances") or {}).items():
        if imp == 0:
            continue
        shown = f"{imp:.4f}" if isinstance(imp, float) else f"{int(imp):6d}"
        print(f"  fi[{fi_type}] {shown:>8}  {name}")
    print(f"[train] backend={info['model_backend']} device={info['device']} "
          f"trees={info.get('trees_used')} fit={info.get('fit_seconds')}s "
          f"total={params['total_seconds']}s")
    return params


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-rows", type=int, default=80_000_000)
    ap.add_argument("--max-eval-rows", type=int, default=None)
    ap.add_argument("--model-type", default=None, choices=["auto", "xgboost", "lightgbm", "sklearn"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config()
    if args.model_type:
        cfg.train.model_type = args.model_type
    if args.device:
        cfg.train.device = args.device
    if args.n_jobs:
        cfg.train.n_jobs = args.n_jobs
    if args.no_gpu:
        cfg.train.use_gpu = False
    run_train(args.split, cfg, max_rows=args.max_rows, max_eval_rows=args.max_eval_rows)


if __name__ == "__main__":
    main()
