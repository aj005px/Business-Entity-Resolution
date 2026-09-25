"""TSV / parquet I/O helpers.

All heavy intermediate data lives in Apache Parquet under ``artifacts/{split}/``;
raw competition TSVs are read column-wise with the pyarrow engine (fast) and kept
as strings (``str`` dtype, no NA coercion so empty address stays empty).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import ARTIFACT_DIR, DATA_DIR, SOURCE_FILES

_TAB = "\t"


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_path(split: str, src: int) -> Path:
    return DATA_DIR / split / SOURCE_FILES[(split, src)]


def artifact_path(split: str, name: str) -> Path:
    return ensure_dir(ARTIFACT_DIR / split) / name


def read_source_tsv(split: str, src: int, **kwargs) -> pd.DataFrame:
    """Read one source TSV as strings (dtype=str keeps leading zeros etc.)."""
    cols = ["entity_id", "business_name", "business_address", "country"]
    path = source_path(split, src)
    kwargs.setdefault("sep", _TAB)
    kwargs.setdefault("dtype", str)
    kwargs.setdefault("keep_default_na", False)
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, usecols=cols, **kwargs)


def read_source_ids(split: str, src: int) -> set[str]:
    """Fast set of entity_id column values (skips pandas)."""
    path = source_path(split, src)
    ids: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if line.strip():
                ids.add(line.split(_TAB, 1)[0].strip())
    return ids


def save_df(df: pd.DataFrame, path) -> None:
    df.to_parquet(path, index=False)


def load_df(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_pyarrow_table(df: pd.DataFrame):
    import pyarrow as pa

    return pa.Table.from_pandas(df, preserve_index=False)


def write_parquet_append(chunk: pd.DataFrame, path) -> None:
    """Append a DataFrame chunk to a parquet file (schema checked)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    table = pa.Table.from_pandas(chunk, preserve_index=False, schema=_schema_from_first(chunk, path))
    if path.exists():
        existing = pq.read_schema(path)
        table = table.cast(existing)
    with pq.ParquetWriter(path, table.schema) as writer:
        writer.write_table(table)


def _schema_from_first(chunk: pd.DataFrame, path: Path):
    """Return the first writer's schema so later chunks match column dtype."""
    import pyarrow as pa

    if path.exists():
        return None  # cast against existing schema instead
    return pa.Table.from_pandas(chunk, preserve_index=False).schema