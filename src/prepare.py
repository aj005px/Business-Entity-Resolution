"""Data preparation (normalize raw TSVs into parquet caches).

Steps for a split (train|test):
    1. Build a stable country -> int mapping from all three sources.
    2. Read each source TSV in chunks, normalize name/address in parallel
       workers, append to ``artifacts/{split}/source{1,2,3}.parquet``.
    3. Concatenate source2 + source3 into ``refs.parquet`` (row index == ref
       position used as the internal reference ID).

Row order is fixed by the source files, so internal positions are stable and
independent of run order -- ids are resolved to positions by array index.
"""
from __future__ import annotations

import argparse
import gc
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pyarrow import parquet as pq
import pyarrow as pa

from .config import Config, NormalizeConfig
from .data_io import ensure_dir, source_path
from .normalize import normalize_address, normalize_name, set_config
from .config import ARTIFACT_DIR

CHUNK = 500_000


def country_mapping(split: str) -> dict[str, int]:
    """Stable int codes for the country column across a whole split."""
    from collections import Counter

    c: Counter[str] = Counter()
    for src in (1, 2, 3):
        path = source_path(split, src)
        for part in pd.read_csv(path, sep="\t", usecols=["country"], dtype=str,
                                chunksize=CHUNK, keep_default_na=False):
            c.update(part["country"].str.strip())
    return {k: i for i, k in enumerate(sorted(c))}


def _norm_worker(payload):
    """(name, address) -> (norm_name, norm_addr); runs in a worker process."""
    name, addr = payload
    return normalize_name(name), normalize_address(addr)


def prepare_source(split: str, src: int, cmap: dict[str, int], workers: int = 16) -> None:
    out_dir = ensure_dir(ARTIFACT_DIR / split)
    out_path = out_dir / f"source{src}.parquet"
    path = source_path(split, src)
    cols = {"entity_id": "entity_id", "business_name": "business_name",
            "business_address": "business_address", "country": "country"}
    first = True
    writer = None
    with Pool(workers, initializer=_init_normalizer, initargs=()) as pool:
        reader = pd.read_csv(path, sep="\t", usecols=list(cols), dtype=str,
                             keep_default_na=False, low_memory=False, chunksize=CHUNK)
        for chunk_ix, df in enumerate(reader):
            df = df.astype(str)
            cols_l = df["business_name"].tolist(), df["business_address"].tolist()
            norms = list(pool.imap(_norm_worker, zip(cols_l[0], cols_l[1]), chunksize=2000))
            df["norm_name"] = [n[0] for n in norms]
            df["norm_addr"] = [n[1] for n in norms]
            df["country_int"] = df["country"].str.strip().map(cmap).fillna(-1).astype(np.int8)
            df["source"] = np.int8(src)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if first:
                writer = pq.ParquetWriter(out_path, table.schema)
                first = False
            writer.write_table(table)
            print(f"  [prepare {split}] {src} chunk {chunk_ix} {len(df)} rows", flush=True)
            del df, norms, table, cols_l
    if writer is not None:
        writer.close()


def _init_normalizer():
    set_config(NormalizeConfig())


def run_prepare(split: str, cfg: Config, workers: int = 16) -> None:
    print(f"[prepare {split}] country mapping ...")
    cmap = country_mapping(split)
    print(f"[prepare {split}] countries: {cmap}")
    for src in (1, 2, 3):
        out_path = ARTIFACT_DIR / split / f"source{src}.parquet"
        if out_path.exists():
            print(f"[prepare {split}] skipping existing {out_path}")
            continue
        print(f"[prepare {split}] normalizing source{src} ...")
        prepare_source(split, src, cmap, workers=workers)
    refs = pd.concat(
        [
            pd.read_parquet(ARTIFACT_DIR / split / "source2.parquet"),
            pd.read_parquet(ARTIFACT_DIR / split / "source3.parquet"),
        ],
        ignore_index=True,
    )
    refs = refs[["entity_id", "norm_name", "norm_addr", "country", "country_int", "source"]]
    refs.to_parquet(ARTIFACT_DIR / split / "refs.parquet", index=False)
    queries = pd.read_parquet(ARTIFACT_DIR / split / "source1.parquet")
    print(f"[prepare {split}] queries={len(queries)} refs={len(refs)}")
    del refs, queries
    gc.collect()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    run_prepare(args.split, Config(), workers=args.workers)


if __name__ == "__main__":
    main()