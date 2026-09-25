"""Central configuration and paths for the entity-resolution pipeline.

Everything is relative to the project root
``code/business_entity_resolution/`` so the whole pipeline can be moved as a
folder and re-run with the same commands.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT / "dataset"))
ARTIFACT_DIR = Path(os.environ.get("ER_ARTIFACT_DIR", ROOT / "artifacts"))
OUTPUT_DIR = ROOT / "output"

TEAM_NAME = "Power Puff Girls"

# Fixed column names used everywhere in the pipeline.
SOURCES = {
    1: "source1",
    2: "source2",
    3: "source3",
}
SOURCE_FILES = {
    (split, src): f"{split}_{SOURCES[src]}.tsv" for split in ("train", "test") for src in (1, 2, 3)
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"

# Reference columns index: which source does each reference row belong to.
# refs.parquet stores (entity_id, norm_name, norm_addr, country_int, source) with
# rows ordered source2-then-source3; internal positions are used as IDs.


@dataclass
class NormalizeConfig:
    """Text-normalization settings (step 2)."""

    min_token_len: int = 2          # drop 1-char tokens from token features/blocking
    strip_combining: bool = True    # NFKD + drop diacritics (Énterprises -> enterprises)
    use_digits: bool = True         # keep numeric tokens (important in addresses)
    name_max_tokens: int = 24       # cap tokens used per record (memory bound)


@dataclass
class BlockingConfig:
    """Candidate-generation settings (step 3). All values are tunable."""

    # Country exact match is a hard filter: empirically 0 ground-truth pairs
    # cross countries, so it costs nothing in recall and halves the search space.
    use_country_block: bool = True

    # --- token blocking ---
    use_name_tokens: bool = True
    use_addr_tokens: bool = True
    max_df_frac: float = 0.02        # block on tokens whose df <= 2% of refs
    relax_df_frac: float = 0.05      # second pass threshold for "generic" queries
    min_block_idf: float = 1.5
    max_tokens_per_query: int = 6    # most discriminative tokens used for blocking
    relax_tokens_per_query: int = 10

    # --- TF-IDF cosine nearest-neighbour blocking ---
    use_name_nn: bool = True
    use_addr_nn: bool = True
    top_k_name: int = 10             # top-k NN candidates added per query (name)
    top_k_addr: int = 10             # top-k NN candidates added per query (address)
    nn_similarity_threshold: float = 0.40   # min cosine kept when pruning by score

    # --- candidate set sizing ---
    min_candidates_per_query: int = 10      # widen tokens until at least this many
    max_candidates_per_query: int = 120     # cap applied only when similarity is low

    # --- TF-IDF ---
    sublinear_tf: bool = True        # tf = 1 + log(count)
    norm_tfidf: bool = True          # unit-length rows (cosine = dot product)

    # chunking / backend
    chunk_size: int = 20_000
    use_sklearn_nn: bool = True      # use sklearn NearestNeighbors on small corpora
    sklearn_nn_max_refs: int = 100_000

    seed: int = 42


@dataclass
class FeatureConfig:
    """Which similarity features are computed per (source1, candidate) pair."""

    use_name_lev: bool = True
    use_addr_lev: bool = True
    use_name_token_jaccard: bool = True
    use_addr_token_jaccard: bool = True
    use_name_token_overlap: bool = True     # |q & r| / |q|
    use_addr_token_overlap: bool = True
    use_name_tfidf_cos: bool = True
    use_addr_tfidf_cos: bool = True
    use_name_trigram: bool = True           # char-trigram Dice similarity
    use_addr_trigram: bool = True
    use_len_diff: bool = True               # name & address char-length gaps
    use_country_match: bool = True
    use_same_source: bool = False           # is the candidate from the same source?
    use_name_char_len: bool = True          # raw char length features for the model

    @property
    def feature_columns(self) -> list[str]:
        cols = []
        if self.use_name_lev:
            cols += ["name_lev"]
        if self.use_addr_lev:
            cols += ["addr_lev"]
        if self.use_name_token_jaccard:
            cols += ["name_token_jaccard"]
        if self.use_addr_token_jaccard:
            cols += ["addr_token_jaccard"]
        if self.use_name_token_overlap:
            cols += ["name_token_overlap"]
        if self.use_addr_token_overlap:
            cols += ["addr_token_overlap"]
        if self.use_name_tfidf_cos:
            cols += ["name_tfidf_cos"]
        if self.use_addr_tfidf_cos:
            cols += ["addr_tfidf_cos"]
        if self.use_name_trigram:
            cols += ["name_trigram"]
        if self.use_addr_trigram:
            cols += ["addr_trigram"]
        if self.use_len_diff:
            cols += ["name_len_diff", "addr_len_diff"]
        if self.use_country_match:
            cols += ["country_match"]
        if self.use_same_source:
            cols += ["same_source"]
        if self.use_name_char_len:
            cols += ["name_char_len_q", "name_char_len_r", "addr_char_len_q", "addr_char_len_r"]
        return cols


@dataclass
class TrainConfig:
    """Model training settings (step 7)."""

    model_type: str = "auto"           # auto | lightgbm | xgboost | sklearn
    val_frac: float = 0.20             # validation split by source1_entity_id
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    subsample_neg_per_pos: float = 2.0  # negatives sampled per positive *per s1*
    max_train_s1: int = 0               # 0 = use all source1 entities
    n_jobs: int = 16
    early_stopping_rounds: int = 50
    seed: int = 42


@dataclass
class TuneConfig:
    """Threshold tuning for the F0.5 metric (step 8)."""

    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01


@dataclass
class Config:
    split: str = "train"
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)


CONFIG = Config()