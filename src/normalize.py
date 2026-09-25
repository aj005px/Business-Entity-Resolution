"""Text normalization for noisy business names/addresses (pipeline step 2).

Strategy (pure local, no external services):
  1. Unicode NFKD decomposition + strip combining marks.
     * ``Énterprises`` -> ``Enterprises`` (accented Latin noise)
     * removes Devanagari/Gujarati nukta marks so both spellings align
  2. Lowercase.
  3. Expand common business/address abbreviations dict-style (corp ->         corporation,
     st -> street, & -> and, ...) so "corp" and "corporation" share a token.
  4. Replace punctuation with spaces (`&` handled before), collapse whitespace.
  5. Tokenize on unicode word characters; optionally drop single-char tokens.

Everything is deterministic and offline -- no transliteration or geo APIs.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .config import NormalizeConfig

# Map every abbreviation to its canonical expanded form. Applied to BOTH names
# and addresses so e.g. "st" vs "street" compares equal. Kept conservative to
# avoid over-merging (e.g. we do NOT map "co" -> "company" because it appears
# in non-company words like "coffee").
ABBREVIATIONS = {
    # US / India corporate suffixes
    "corp": "corporation",
    "corps": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "ltd.": "limited",
    "pvt": "private",
    "llp": "limited liability partnership",
    "pllc": "professional limited liability company",
    # US postal suffixes
    "st": "street",
    "st.": "street",
    "rd": "road",
    "rd.": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "ln.": "lane",
    "dr": "drive",
    "hwy": "highway",
    "pkwy": "parkway",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "trl": "trail",
    "mt": "mount",
    "mt.": "mount",
    # Indian suffixes (English)
    "nagar": "nagar",
    "chowk": "chowk",
}


@lru_cache(maxsize=1_000_000)
def _expand_cache(text: str) -> str:
    """Cache abbreviation expansion for repeated strings (huge speedup)."""
    out = []
    for word in text.split():
        out.append(ABBREVIATIONS.get(word, word))
    return " ".join(out)


def clean_text(value) -> str:
    """Core cleaning: NFKD, strip combining marks, lowercase, punct -> space.

    Returns a whitespace-normalized string (no tokens, no abbreviations).
    """
    if value is None:
        return ""
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    if _cfg_strip_combining():
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # & is punctuation replaced with the word "and" BEFORE the punct strip.
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", " ", s)  # zero-width markers
    return " ".join(s.split())


# small indirection so the cache key stays cheap (read config once)
_CONFIG: NormalizeConfig | None = None


def _cfg_strip_combining() -> bool:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = NormalizeConfig()
    return _CONFIG.strip_combining


def set_config(cfg: NormalizeConfig) -> None:
    global _CONFIG
    _CONFIG = cfg


def normalize_name(value) -> str:
    """Normalize a business name (cleaning + abbreviation expansion)."""
    return _expand_cache(clean_text(value))


def normalize_address(value) -> str:
    """Normalize a business address (cleaning + abbreviation expansion)."""
    return _expand_cache(clean_text(value))


def tokenize(text: str, min_len: int | None = None) -> list[str]:
    """Split normalized text into tokens (word chars, min length filter)."""
    if min_len is None:
        min_len = _cfg_min_len()
    return [tok for tok in text.split() if len(tok) >= min_len] if text else []


def _cfg_min_len() -> int:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = NormalizeConfig()
    return _CONFIG.min_token_len


def normalize_and_tokenize(value, min_len: int | None = None):
    """One-stop: clean + expand + tokenize -> (normalized_str, tokens)."""
    norm = normalize_name(value)  # same expansion table for names and addresses
    return norm, tokenize(norm, min_len)


def char_trigrams(text: str) -> frozenset[str]:
    """Character 3-grams (padding with spaces) used for typos / fuzzy hashing."""
    if not text:
        return frozenset()
    body = " " + text + " "
    return frozenset(body[i : i + 3] for i in range(len(body) - 2))


def dice(a, b) -> float:
    """Dice coefficient = 2*|A∩B| / (|A|+|B|) over two sets/sequences.

    Used with character trigram sets to be robust to spelling typos.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    return 2.0 * inter / (len(a) + len(b))