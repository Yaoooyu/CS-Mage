"""Compute raw and normalized Chinese ASR metrics from prediction CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from normalize_text import normalize_text, tokenize_for_wer


def edit_counts(reference: list[str], hypothesis: list[str]) -> Counter:
    table = [[(0, 0, 0, 0)] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1): table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hypothesis) + 1): table[0][j] = (j, 0, 0, j)
    for i, source in enumerate(reference, 1):
        for j, target in enumerate(hypothesis, 1):
            if source == target:
                table[i][j] = table[i - 1][j - 1]
            else:
                candidates = [(table[i - 1][j][0] + 1, 0, 1, 0), (table[i][j - 1][0] + 1, 0, 0, 1), (table[i - 1][j - 1][0] + 1, 1, 0, 0)]
                distance, s, d, ins = min(candidates, key=lambda item: item[0])
                prev = table[i - 1][j - 1] if s else table[i - 1][j] if d else table[i][j - 1]
                table[i][j] = (distance, prev[1] + s, prev[2] + d, prev[3] + ins)
    _, substitutions, deletions, insertions = table[-1][-1]
    return Counter(substitutions=substitutions, deletions=deletions, insertions=insertions, errors=substitutions + deletions + insertions)


def raw_word_tokens(text: object) -> list[str]:
    try:
        import jieba  # type: ignore
        return [token for token in jieba.lcut(str(text)) if token.strip()]
    except Exception:
        return list(str(text))


def score(rows: list[dict], normalized: bool, word_level: bool = False) -> dict:
    aggregate, ref_units, sent_errors = Counter(), 0, 0
    for row in rows:
        ref, hyp = row["reference_text"], row["prediction"]
        if normalized: ref, hyp = normalize_text(ref), normalize_text(hyp)
        if word_level:
            units_ref = tokenize_for_wer(ref) if normalized else raw_word_tokens(ref)
            units_hyp = tokenize_for_wer(hyp) if normalized else raw_word_tokens(hyp)
        else:
            units_ref, units_hyp = list(ref), list(hyp)
        counts = edit_counts(units_ref, units_hyp)
        aggregate.update(counts); ref_units += len(units_ref); sent_errors += counts["errors"] > 0
    return {"error_rate": aggregate["errors"] / max(1, ref_units), "substitutions": aggregate["substitutions"], "deletions": aggregate["deletions"], "insertions": aggregate["insertions"], "reference_units": ref_units, "sentence_error_rate": sent_errors / max(1, len(rows))}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--metrics", type=Path, required=True); parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args(); started = time.time()
    with args.predictions.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    normalized_char_rows = [{**r, "reference_text": normalize_text(r["reference_text"]), "prediction": normalize_text(r["prediction"])} for r in rows]
    cer = score(normalized_char_rows, normalized=False)
    wer = score(rows, normalized=True, word_level=True)
    raw_char = score(rows, normalized=False)
    raw_wer = score(rows, normalized=False, word_level=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata.update({"CER": cer["error_rate"], "WER": wer["error_rate"], "sentence_error_rate": cer["sentence_error_rate"], "substitutions": cer["substitutions"], "deletions": cer["deletions"], "insertions": cer["insertions"], "raw_CER": raw_char["error_rate"], "raw_WER": raw_wer["error_rate"], "n_samples": len(rows), "normalization": "NFKC; lowercase Latin; retain Han/digits/a-z; remove punctuation and whitespace", "evaluation_seconds": time.time() - started, "status": "completed"})
    args.metrics.parent.mkdir(parents=True, exist_ok=True); args.metrics.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__": main()
