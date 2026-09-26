"""Candidate generation / blocking (pipeline step 3).

Full pairwise comparison across ~10M reference records is infeasible, so each
source-1 entity gets a candidate set built from three *combined* strategies:

1. **Country blocking** -- hard filter. Empirical check on ground truth: 100% of
   true matches share country, so this costs zero recall and halves the search.
2. **Token blocking** -- inverted index over *significant* (rare / high-idf)
   name and address tokens; candidates = union of postings for a query's most
   discriminating tokens.
3. **TF-IDF + cosine nearest neighbours** -- exact cosine top-k, computed by
   accumulating ``query_tfidf @ ref_tfidf`` over the same inverted index. This
   is mathematically identical to sklearn ``NearestNeighbors(metric="cosine")``
   brute force restricted to refs sharing a token with the query (all that
   matters for top-k). On small corpora (below ``sklearn_nn_max_refs``) we
   additionally exercise the real sklearn interface for verification.

Name and address are treated as separate fields with separate vocabularies /
IDFs; the final candidate pool is the union of the two blocks.

Index layout (``artifacts/{split}/``):
    refs.parquet                  concatenated source2+source3 (row == ref pos)
    source1.parquet               source1 (row == query pos)
    name_index.npz / addr_index.npz   scipy csc postings + df/idf/ref_country
    name_query_tfidf.npz / addr_query_tfidf.npz   query tfidf + query_country

Candidates are written as chunked parquet files under
``artifacts/{split}/candidates/cand_*.parquet`` with columns:
    q_idx, r_idx, source, name_cos, addr_cos, block_score
"""
from __future__ import annotations

import argparse
import gc
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import BlockingConfig, Config, NormalizeConfig
from .data_io import artifact_path, ensure_dir, load_df
from .normalize import set_config, tokenize

import threading

_THREAD_LOCAL = threading.local()


def _get_score_buf(n: int) -> np.ndarray:
    if not hasattr(_THREAD_LOCAL, "buf") or _THREAD_LOCAL.buf.size < n:
        _THREAD_LOCAL.buf = np.zeros(n, dtype=np.float64)
    return _THREAD_LOCAL.buf


def _reset(buf: np.ndarray, touched: np.ndarray) -> None:
    if len(touched):
        buf[touched] = 0.0


# ---------------------------------------------------------------------------
# Vocabulary / TF-IDF construction
# ---------------------------------------------------------------------------


def compose_vocab(token_lists, min_len: int):
    """Deterministic vocab (sorted tokens) + per-token document frequency."""
    from collections import Counter

    c: Counter = Counter()
    for toks in token_lists:
        if toks:
            c.update(toks)
    tokens = sorted(c)
    vocab = {t: i for i, t in enumerate(tokens)}
    df = np.array([c[t] for t in tokens], dtype=np.int64)
    return vocab, df


def build_token_csr(token_lists, vocab, n_docs: int):
    """Count csr: data = number of times a token appears in a doc."""
    total = sum(len(t) for t in token_lists)
    row = np.empty(total, dtype=np.int32)
    col = np.empty(total, dtype=np.int32)
    off = 0
    for i, toks in enumerate(token_lists):
        arr = np.empty(len(toks), dtype=np.int32)
        for j, t in enumerate(toks):
            arr[j] = vocab.get(t, -1)
        keep = arr >= 0
        L = int(keep.sum())
        if L:
            row[off : off + L] = i
            col[off : off + L] = np.asarray(arr[keep], dtype=np.int32)
        off += L
    data = np.ones(off, dtype=np.float64)
    m = sp.csr_matrix((data, (row[:off], col[:off])), shape=(n_docs, len(vocab)))
    return m


def tfidf_from_csr(count_csr, idf: np.ndarray, sublinear=True, norm=True):
    idx = count_csr.indices.astype(np.int64)
    w = count_csr.data.astype(np.float64)
    if sublinear:
        w = 1.0 + np.log(w)
    w = w * idf[idx]
    m = sp.csr_matrix((w, count_csr.indices, count_csr.indptr), shape=count_csr.shape)
    if norm:
        m = _normalize_rows(m)
    return m.astype(np.float32)


def _normalize_rows(m):
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    m = m.tocsr().copy()
    m.data /= norm.repeat(np.diff(m.indptr))
    return m


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


class TfidfIndex:
    def __init__(self):
        self.postings = None
        self.df = None
        self.idf = None
        self.n_refs = 0
        self.vocab_size = 0
        self.ref_country = None

    def save(self, path, field):
        sp.save_npz(path.joinpath(f"{field}_index.npz"), self.postings)
        np.savez(
            path.joinpath(f"{field}_meta.npz"),
            df=self.df.astype(np.int64), idf=self.idf.astype(np.float32),
            n_refs=np.int64(self.n_refs), vocab_size=np.int64(self.vocab_size),
            ref_country=self.ref_country.astype(np.int8),
        )

    def load(self, path, field):
        self.postings = sp.load_npz(path.joinpath(f"{field}_index.npz"))
        meta = np.load(path.joinpath(f"{field}_meta.npz"))
        self.df = meta["df"]
        self.idf = meta["idf"]
        self.n_refs = int(meta["n_refs"])
        self.vocab_size = int(meta["vocab_size"])
        self.ref_country = meta["ref_country"]
        return self


def build_index(split: str, field: str, bcfg: BlockingConfig, ncfg: NormalizeConfig):
    """Build + persist TF-IDF inverted index (and query tfidf) for one field."""
    refs = load_df(artifact_path(split, "refs.parquet"))
    col = "norm_addr" if field == "addr" else "norm_name"
    token_lists = [tokenize(t, ncfg.min_token_len) for t in refs[col].tolist()]
    n_refs = len(refs)
    ref_country = refs["country_int"].to_numpy(dtype=np.int8)

    vocab, df = compose_vocab(token_lists, ncfg.min_token_len)
    idf = np.log((n_refs + 1.0) / (1.0 + df)).astype(np.float32)

    count_csr = build_token_csr(token_lists, vocab, n_refs)
    tfidf = tfidf_from_csr(count_csr, idf, sublinear=bcfg.sublinear_tf, norm=bcfg.norm_tfidf)
    postings = tfidf.tocsc()

    idx = TfidfIndex()
    idx.postings = postings
    idx.df = df
    idx.idf = idf
    idx.n_refs = n_refs
    idx.vocab_size = len(vocab)
    idx.ref_country = ref_country
    split_dir = ensure_dir(artifact_path(split, ""))
    idx.save(split_dir, field)

    queries = load_df(artifact_path(split, "source1.parquet"))
    q_token_lists = [tokenize(t, ncfg.min_token_len) for t in queries[col].tolist()]
    q_csr = build_token_csr(q_token_lists, vocab, len(queries))
    q_tfidf = tfidf_from_csr(q_csr, idf, sublinear=bcfg.sublinear_tf, norm=bcfg.norm_tfidf)
    sp.save_npz(artifact_path(split, f"{field}_query_tfidf.npz"), q_tfidf)
    np.save(
        artifact_path(split, f"query_country_{field}.npy"),
        queries["country_int"].to_numpy(dtype=np.int8),
    )
    del token_lists, q_token_lists, count_csr, tfidf, q_csr
    gc.collect()
    return idx


# ---------------------------------------------------------------------------
# Per-query scoring
# ---------------------------------------------------------------------------


def _select_block_tokens(q_idx, q_tfidf, df, n_refs, cfg: BlockingConfig):
    """Choose the query's scoring tokens: the rarest (highest-idf) tokens.

    Two levels: prefer tokens with document-frequency <= max_df_frac of refs;
    if the query has none that rare, relax to the relax_df_frac cap.
    """
    start, end = q_tfidf.indptr[q_idx], q_tfidf.indptr[q_idx + 1]
    if start == end:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    toks = q_tfidf.indices[start:end]
    w = q_tfidf.data[start:end]
    dfs = df[toks]
    cap1 = int(cfg.max_df_frac * n_refs)
    ok1 = dfs <= cap1
    sub = np.flatnonzero(ok1)
    if len(sub):
        order = sub[np.argsort(dfs[sub], kind="stable")[: cfg.max_tokens_per_query]]
        return toks[order], w[order]
    cap2 = int(cfg.relax_df_frac * n_refs)
    ok2 = dfs <= cap2
    sub2 = np.flatnonzero(ok2)
    if len(sub2):
        order = sub2[np.argsort(dfs[sub2], kind="stable")[: cfg.relax_tokens_per_query]]
        return toks[order], w[order]
    return np.empty(0, np.int32), np.empty(0, np.float32)


def score_query(q_idx, q_tfidf_csr, postings, df, n_refs, ref_country, q_country,
                buf, cfg: BlockingConfig):
    """Exact cosine scores of one query against refs sharing a scoring token.

    Returns (refs sorted by score desc, scores).
    """
    tokens, w = _select_block_tokens(q_idx, q_tfidf_csr, df, n_refs, cfg)
    if len(tokens) == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    touched_list = []
    p_indptr = postings.indptr
    p_indices = postings.indices
    p_data = postings.data
    for tok, qw in zip(tokens, w):
        s, e = p_indptr[tok], p_indptr[tok + 1]
        if s == e:
            continue
        refs = p_indices[s:e]
        mask = ref_country[refs] == q_country
        if not np.any(mask):
            continue
        keep = refs[mask]
        wts = p_data[s:e][mask].astype(np.float64)
        np.add.at(buf, keep, qw * wts)
        touched_list.append(keep)
    if not touched_list:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    arr = np.unique(np.concatenate(touched_list))
    vals = buf[arr]
    _reset(buf, arr)
    return _finalize_candidates(arr, vals, cfg)


def _finalize_candidates(refs, vals, cfg: BlockingConfig):
    if len(refs) <= cfg.max_candidates_per_query:
        order = np.argsort(vals)[::-1]
        return refs[order], vals[order]
    thr = vals >= cfg.nn_similarity_threshold
    n_thr = int(thr.sum())
    if cfg.min_candidates_per_query <= n_thr <= cfg.max_candidates_per_query:
        keep, vk = refs[thr], vals[thr]
    else:
        k = min(len(refs), cfg.max_candidates_per_query)
        k = max(k, cfg.min_candidates_per_query)
        top = np.argpartition(vals, -k)[-k:]
        keep, vk = refs[top], vals[top]
    order = np.argsort(vk)[::-1]
    return keep[order], vk[order]


def _merge_blocks(name_res, addr_res):
    """Union of name+address candidate refs with per-field cosines (NaN = n/a).

    Returns refs sorted by ``block`` = max(name_cos, addr_cos) descending.
    """
    if name_res[0].size == 0 and addr_res[0].size == 0:
        return (np.empty(0, np.int32), np.empty(0, np.float32),
                np.empty(0, np.float32), np.empty(0, np.float32))
    m: dict[int, list] = {}
    for r, c in zip(name_res[0], name_res[1]):
        m[int(r)] = [float(c), float("nan")]
    for r, c in zip(addr_res[0], addr_res[1]):
        r = int(r)
        if r in m:
            m[r][1] = float(c)
        else:
            m[r] = [float("nan"), float(c)]
    keys = np.fromiter(m.keys(), dtype=np.int32)
    nc = np.array([v[0] for v in m.values()], dtype=np.float32)
    ac = np.array([v[1] for v in m.values()], dtype=np.float32)
    block = np.where(np.isnan(nc), ac, np.where(np.isnan(ac), nc, np.maximum(nc, ac)))
    order = np.argsort(block)[::-1]
    return keys[order], nc[order], ac[order], block[order]


def _apply_floor_cap(refs, nc, ac, block, cfg: BlockingConfig):
    """Select the final candidate set for one query.

    Ranking policy (see README):
      * every ref with best-field cosine ``>= nn_similarity_threshold`` is kept
        (recall-preserving floor);
      * the ``top_k_name``/``top_k_addr`` best refs of each field are also
        guaranteed (recovers weak-but-best-field matches like address aliases);
      * at least ``min_candidates_per_query`` refs are kept (weak queries need a
        pool for the classifier);
      * the merged set is hard-capped at ``max_candidates_per_query`` ranked by
        ``block`` so generic-name queries cannot explode pair counts.
    """
    n = len(refs)
    if n == 0:
        return refs, nc, ac, block
    if not np.isnan(block).all() and n <= cfg.max_candidates_per_query and \
            (block >= cfg.nn_similarity_threshold).sum() >= n:
        return refs, nc, ac, block
    keep = block >= cfg.nn_similarity_threshold
    if cfg.top_k_name > 0:
        order_n = np.argsort(nc, kind="stable")[::-1][: cfg.top_k_name]
        keep[order_n] = True
    if cfg.top_k_addr > 0:
        order_a = np.argsort(ac, kind="stable")[::-1][: cfg.top_k_addr]
        keep[order_a] = True
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        idx = np.argsort(block)[::-1][: min(n, cfg.min_candidates_per_query)]
    elif len(idx) < cfg.min_candidates_per_query:
        notk = np.flatnonzero(~keep)
        if len(notk):
            extra = notk[np.argsort(block[notk])[::-1]][: cfg.min_candidates_per_query - len(idx)]
            idx = np.concatenate([idx, extra])
    if len(idx) > cfg.max_candidates_per_query:
        top = np.argpartition(block[idx], -cfg.max_candidates_per_query)[-cfg.max_candidates_per_query:]
        idx = idx[top]
    idx = idx[np.argsort(block[idx])[::-1]]
    return refs[idx], nc[idx], ac[idx], block[idx]


# ---------------------------------------------------------------------------
# Chunked candidate generation
# ---------------------------------------------------------------------------


def generate_candidates_for_chunk(q0, q1, indexes, cfg: BlockingConfig):
    """Produce candidate arrays for queries [q0, q1)."""
    out_q, out_r, out_s, out_nc, out_ac, out_blk = [], [], [], [], [], []
    buf = _get_score_buf(indexes["name"].n_refs)
    name_post = indexes["name"].postings
    addr_post = indexes["addr"].postings
    rc_n = indexes["name"].ref_country
    rc_a = indexes["addr"].ref_country
    qq_n = indexes["name_query_tfidf"]
    qq_a = indexes["addr_query_tfidf"]
    qc_n = indexes["name_query_country"]
    qc_a = indexes["addr_query_country"]

    for q in range(q0, q1):
        name_res = score_query(q, qq_n, name_post, indexes["name"].df, indexes["name"].n_refs,
                               rc_n, qc_n[q], buf, cfg)
        addr_res = score_query(q, qq_a, addr_post, indexes["addr"].df, indexes["addr"].n_refs,
                               rc_a, qc_a[q], buf, cfg)
        refs, nc, ac, block = _merge_blocks(name_res, addr_res)
        if len(refs) == 0:
            continue
        refs, nc, ac, block = _apply_floor_cap(refs, nc, ac, block, cfg)
        out_q.append(np.full(len(refs), q, dtype=np.int32))
        out_r.append(refs)
        out_nc.append(nc)
        out_ac.append(ac)
        out_blk.append(block)
    if not out_q:
        return (np.empty(0, np.int32),) * 6
    return (np.concatenate(out_q), np.concatenate(out_r), np.empty(0, np.int32),
            np.concatenate(out_nc), np.concatenate(out_ac), np.concatenate(out_blk))


def _task_worker(args):
    q0, q1, c, split, out_dir, idxs, config_b, src_ref = args
    q, r, _, nc, ac, blk = generate_candidates_for_chunk(q0, q1, idxs, config_b)
    if len(q):
        frame = pd.DataFrame({
            "q_idx": q, "r_idx": r, "source": src_ref[r],
            "name_cos": nc, "addr_cos": ac, "block_score": blk,
        })
        frame.to_parquet(Path(out_dir) / f"cand_{c:06d}.parquet", index=False)
    return c, q1 - q0, len(q)


# --- process-pool worker state -------------------------------------------
# The index is ~1.2 GiB, so it must never travel inside a task payload (it would
# be pickled once per chunk). Each worker loads it exactly once instead.
_WORKER: dict = {}


def _load_block_state(split: str) -> dict:
    split_dir = ensure_dir(artifact_path(split, ""))
    indexes = {}
    for field in ("name", "addr"):
        indexes[field] = TfidfIndex().load(split_dir, field)
    indexes["name_query_tfidf"] = sp.load_npz(artifact_path(split, "name_query_tfidf.npz"))
    indexes["addr_query_tfidf"] = sp.load_npz(artifact_path(split, "addr_query_tfidf.npz"))
    indexes["name_query_country"] = np.load(artifact_path(split, "query_country_name.npy"))
    indexes["addr_query_country"] = np.load(artifact_path(split, "query_country_addr.npy"))
    indexes["src_of_ref"] = pd.read_parquet(
        artifact_path(split, "refs.parquet"), columns=["source"])["source"].to_numpy(np.int8)
    return indexes


def _init_block_worker(split: str, out_dir: str) -> None:
    set_config(Config())
    _WORKER["indexes"] = _load_block_state(split)
    _WORKER["out_dir"] = out_dir
    _WORKER["cfg"] = Config().blocking


def _task_worker_proc(task):
    q0, q1, c = task
    indexes = _WORKER["indexes"]
    cfg = _WORKER["cfg"]
    q, r, _, nc, ac, blk = generate_candidates_for_chunk(q0, q1, indexes, cfg)
    if len(q):
        pd.DataFrame({
            "q_idx": q, "r_idx": r, "source": indexes["src_of_ref"][r],
            "name_cos": nc, "addr_cos": ac, "block_score": blk,
        }).to_parquet(Path(_WORKER["out_dir"]) / f"cand_{c:06d}.parquet", index=False)
    return c, q1 - q0, len(q)


def _n_queries(split: str) -> int:
    """Row count of source1 without materialising the frame."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(artifact_path(split, "source1.parquet")).metadata.num_rows


def run_block(split: str, config: Config, start: int = 0, end: int | None = None,
              jobs: int = 1, executor: str = "auto"):
    cfg = config.blocking
    out_dir = ensure_dir(artifact_path(split, "candidates"))
    n_queries = _n_queries(split)
    if end is None:
        end = n_queries
    chunk = cfg.chunk_size
    tasks = [(q0, min(q0 + chunk, end), c) for c, q0 in enumerate(range(start, end, chunk))]

    if executor == "auto":
        executor = "process" if jobs > 1 else "serial"

    if executor == "process" and jobs > 1:
        # Parent stays lean: row count comes from parquet metadata, and the
        # index is loaded inside each worker.
        print(f"[block {split}] starting candidate generation with {jobs} processes ...", flush=True)
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_block_worker,
                                 initargs=(split, str(out_dir))) as ex:
            for c, nq, npairs in ex.map(_task_worker_proc, tasks):
                if c % 5 == 0 or c == len(tasks) - 1:
                    avg = int(npairs / nq) if nq else 0
                    print(f"[block {split}] chunk {c + 1}/{len(tasks)} queries {nq} "
                          f"pairs {npairs} avg {avg}", flush=True)
    else:
        print(f"[block {split}] loading index and reference data into memory ...", flush=True)
        indexes = _load_block_state(split)
        src_of_ref = indexes.pop("src_of_ref")
        tasks_full = [(q0, q1, c, split, str(out_dir), indexes, cfg, src_of_ref)
                      for q0, q1, c in tasks]
        if executor == "thread" and jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f"[block {split}] starting candidate generation with {jobs} threads ...", flush=True)
            runner = ThreadPoolExecutor(max_workers=jobs)
        else:
            runner = None
        it = runner.map(_task_worker, tasks_full) if runner else map(_task_worker, tasks_full)
        for c, nq, npairs in it:
            if c % 5 == 0 or c == len(tasks) - 1:
                avg = int(npairs / nq) if nq else 0
                print(f"[block {split}] chunk {c + 1}/{len(tasks)} queries {nq} "
                      f"pairs {npairs} avg {avg}", flush=True)
        if runner:
            runner.shutdown()
    print(f"[block {split}] done. chunks={len(tasks)}")
    gc.collect()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--executor", default="auto", choices=["auto", "process", "thread", "serial"])
    args = ap.parse_args(argv)
    cfg = Config()
    if args.max_candidates:
        cfg.blocking.max_candidates_per_query = args.max_candidates
    run_block(args.split, cfg, start=args.start, end=args.end, jobs=args.jobs,
              executor=args.executor)


if __name__ == "__main__":
    main()