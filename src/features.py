"""Pair similarity features (pipeline step 5).

For every (source1, candidate) pair we emit a row of numeric features:
  * Levenshtein ratio on normalized name and address (rapidfuzz, C-accelerated)
  * Jaccard similarity + query-side token overlap on name / address tokens
  * TF-IDF cosine similarity on name and address (computed during blocking --
    ``name_cos`` / ``addr_cos``)
  * character-trigram Dice similarity on name / address (typo robustness)
  * country match (binary) and absolute string-length differences
  * raw character lengths (query/ref pair) for the model

Pairs are streamed chunk-by-chunk from the blocking output and written back out
as ``artifacts/{split}/features/feat_*.parquet``. For the train split a ``label``
column (1 = true match, 0 = not) is joined from ground truth while streaming.
"""
from __future__ import annotations

import argparse
import gc
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import Config, FeatureConfig, TrainConfig
from .data_io import artifact_path, ensure_dir, load_df
from .label import label_pairs, load_label_keys
from .normalize import char_trigrams, dice, set_config, tokenize, NormalizeConfig


def _unique_str_sets(strings, kind: str, ncfg: NormalizeConfig):
    """Cache token sets / trigram sets per unique string."""
    token_cache = {}
    trigram_cache = {}
    for s in strings:
        if s not in token_cache:
            token_cache[s] = set(tokenize(s, ncfg.min_token_len))
        if s not in trigram_cache:
            trigram_cache[s] = char_trigrams(s)
    return token_cache, trigram_cache


def compute_chunk_features(q_name, r_name, q_addr, r_addr, qc, rc, ncfg: NormalizeConfig):
    """Compute all scalar similarity features for a list of pairs.

    q_name/r_name/... are parallel python lists (one entry per pair). Returns a
    dictionary of numpy arrays. Runs inside a worker process.
    """
    n = len(q_name)
    lev_n = np.zeros(n, np.float32)
    lev_a = np.zeros(n, np.float32)
    ja_n = np.zeros(n, np.float32)
    ja_a = np.zeros(n, np.float32)
    ov_n = np.zeros(n, np.float32)
    ov_a = np.zeros(n, np.float32)
    tri_n = np.zeros(n, np.float32)
    tri_a = np.zeros(n, np.float32)
    dif_n = np.zeros(n, np.float32)
    dif_a = np.zeros(n, np.float32)
    clen_qn = np.zeros(n, np.float32)
    clen_rn = np.zeros(n, np.float32)
    clen_qa = np.zeros(n, np.float32)
    clen_ra = np.zeros(n, np.float32)
    cmatch = np.zeros(n, np.float32)

    tok_cache, trigcache = {}, {}
    for i in range(n):
        qn = q_name[i]
        rn = r_name[i]
        qa = q_addr[i]
        ra = r_addr[i]
        if qn not in tok_cache:
            tok_cache[qn] = set(tokenize(qn, ncfg.min_token_len))
        if rn not in tok_cache:
            tok_cache[rn] = set(tokenize(rn, ncfg.min_token_len))
        if qn not in trigcache:
            trigcache[qn] = char_trigrams(qn)
        if rn not in trigcache:
            trigcache[rn] = char_trigrams(rn)
        qt, rt = tok_cache[qn], tok_cache[rn]
        if qn and rn:
            lev_n[i] = fuzz.ratio(qn, rn) / 100.0
            ja_n[i] = dice(qt, rt)
            inter = len(qt & rt)
            ov_n[i] = inter / max(1, len(qt))
            tri_n[i] = dice(trigcache[qn], trigcache[rn])
        if qa and ra:
            if qa not in tok_cache:
                tok_cache[qa] = set(tokenize(qa, ncfg.min_token_len))
            if ra not in tok_cache:
                tok_cache[ra] = set(tokenize(ra, ncfg.min_token_len))
            leva = tok_cache[qa] & tok_cache[ra]
            lev_a[i] = fuzz.ratio(qa, ra) / 100.0
            ja_a[i] = dice(tok_cache[qa], tok_cache[ra])
            ov_a[i] = len(leva) / max(1, len(tok_cache[qa]))
            tri_a[i] = dice(char_trigrams(qa), char_trigrams(ra))
        dif_n[i] = abs(len(qn) - len(rn))
        dif_a[i] = abs(len(qa) - len(ra))
        clen_qn[i] = len(qn)
        clen_rn[i] = len(rn)
        clen_qa[i] = len(qa)
        clen_ra[i] = len(ra)
        cmatch[i] = 1.0 if qc[i] == rc[i] else 0.0
    return {
        "name_lev": lev_n, "addr_lev": lev_a,
        "name_token_jaccard": ja_n, "addr_token_jaccard": ja_a,
        "name_token_overlap": ov_n, "addr_token_overlap": ov_a,
        "name_trigram": tri_n, "addr_trigram": tri_a,
        "name_len_diff": dif_n, "addr_len_diff": dif_a,
        "country_match": cmatch,
        "name_char_len_q": clen_qn, "name_char_len_r": clen_rn,
        "addr_char_len_q": clen_qa, "addr_char_len_r": clen_ra,
    }


def _chunk_payload(q_pos, r_pos, queries, refs, ncfg):
    """Gather per-pair string lists for a worker (pickle-friendly)."""
    q_idx = int(q_pos.shape[0])
    q_name = queries["norm_name"][q_pos].tolist()
    r_name = refs["norm_name"][r_pos].tolist()
    q_addr = queries["norm_addr"][q_pos].tolist()
    r_addr = refs["norm_addr"][r_pos].tolist()
    qc = queries["country_int"][q_pos].tolist()
    rc = refs["country_int"][r_pos].tolist()
    return q_name, r_name, q_addr, r_addr, qc, rc


def run_features(split: str, cfg: Config, start_chunk: int = 0, end_chunk: int | None = None) -> int:
    fcfg = cfg.features
    ncfg = cfg.normalize
    jcfg = cfg.train
    out_dir = ensure_dir(artifact_path(split, "features"))
    queries = load_df(artifact_path(split, "source1.parquet"))
    refs = load_df(artifact_path(split, "refs.parquet"))
    keys = load_label_keys(split) if split == "train" else None

    cand_dir = artifact_path(split, "candidates")
    cand_files = sorted(cand_dir.glob("cand_*.parquet"))
    if end_chunk is None:
        end_chunk = len(cand_files)
    files = cand_files[start_chunk:end_chunk]

    wanted = set(fcfg.feature_columns)
    total_pairs = 0
    slice_per_chunk = max(1, fcfg.pairs_per_task)
    max_inflight = max(1, fcfg.max_inflight)
    nproc = max(1, min(jcfg.n_jobs, max_inflight))
    if nproc < jcfg.n_jobs:
        print(f"[features {split}] {jcfg.n_jobs} jobs capped to {nproc} "
              f"by max_inflight={max_inflight}", flush=True)

    pending: list = []

    def harvest(block: bool = False) -> bool:
        """Write finished slices; blocks on the oldest in-flight task."""
        nonlocal total_pairs
        while pending and (block or len(pending) >= max_inflight):
            async_res, c, s, q_idx, r_idx, source, nc_, ac_, blk = pending.pop(0)
            res = async_res.get()
            out = pd.DataFrame({
                "q_idx": q_idx, "r_idx": r_idx, "source": source,
                "name_tfidf_cos": nc_, "addr_tfidf_cos": ac_, "block_score": blk,
            } | {k: res[k] for k in res if k in wanted})
            if keys is not None:
                out["label"] = label_pairs(out["q_idx"].to_numpy(np.int32),
                                           out["r_idx"].to_numpy(np.int32), keys)
            out.to_parquet(out_dir / f"feat_{c:06d}_{s:03d}.parquet", index=False)
            total_pairs += len(out)
            del out, res
        return True

    with Pool(nproc, initializer=_feat_worker_init) as pool:
        for c, f in enumerate(files):
            cand = pd.read_parquet(f)
            if len(cand) == 0:
                continue
            n_rows = len(cand)
            n_slices = max(1, (n_rows + slice_per_chunk - 1) // slice_per_chunk)
            for s in range(n_slices):
                lo, hi = s * slice_per_chunk, min((s + 1) * slice_per_chunk, n_rows)
                payload = _chunk_payload(
                    cand["q_idx"].to_numpy(np.int32)[lo:hi],
                    cand["r_idx"].to_numpy(np.int32)[lo:hi],
                    queries, refs, ncfg,
                )
                async_res = pool.apply_async(compute_chunk_features,
                                             args=payload + (ncfg,))
                pending.append((async_res, start_chunk + c, s,
                                cand["q_idx"].to_numpy(np.int32)[lo:hi],
                                cand["r_idx"].to_numpy(np.int32)[lo:hi],
                                cand["source"].to_numpy(np.int8)[lo:hi],
                                cand["name_cos"].to_numpy(np.float32)[lo:hi],
                                cand["addr_cos"].to_numpy(np.float32)[lo:hi],
                                cand["block_score"].to_numpy(np.float32)[lo:hi]))
                harvest()
            del cand
            if c % 5 == 0 or c == len(files) - 1:
                print(f"[features {split}] file {c+1}/{len(files)} pairs={total_pairs:,} "
                      f"inflight={len(pending)}", flush=True)
        harvest(block=True)
    print(f"[features {split}] done. total pairs={total_pairs:,}")
    del queries, refs
    gc.collect()
    return total_pairs


def _feat_worker_init():
    set_config(NormalizeConfig())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--end-chunk", type=int, default=None)
    args = ap.parse_args(argv)
    run_features(args.split, Config(), start_chunk=args.start_chunk, end_chunk=args.end_chunk)


if __name__ == "__main__":
    main()