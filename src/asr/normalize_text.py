"""Shared deterministic Chinese ASR text normalization."""
from __future__ import annotations

import re
import unicodedata


def normalize_text(text: object) -> str:
    """NFKC-normalize text and retain CJK, digits, and ASCII letters only."""
    text = unicodedata.normalize("NFKC", "" if text is None else str(text)).lower()
    # Keep Han characters, Arabic digits, and Latin letters. Punctuation,
    # whitespace, emojis, and markup are deliberately removed for every model.
    return "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff0-9a-z]", text))


def tokenize_for_wer(text: object) -> list[str]:
    """Use jieba when available; fall back deterministically to characters."""
    normalized = normalize_text(text)
    try:
        import jieba  # type: ignore
        return [token for token in jieba.lcut(normalized) if token.strip()]
    except Exception:
        return list(normalized)

