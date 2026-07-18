"""Validate CS-Mage ASR resources and create fixed, duration-balanced splits."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import re
import statistics
import wave
import zipfile
from collections import Counter
from pathlib import Path

from normalize_text import normalize_text


def expected_wav(sample_id: str) -> str:
    match = re.fullmatch(r"video_(\d+)_(\d+)", sample_id)
    if not match:
        raise ValueError(f"Unexpected sample id: {sample_id!r}")
    return f"{int(match.group(1)):02d}_{int(match.group(2)):04d}.wav"


def balanced_split(rows: list[dict], seed: int = 42) -> None:
    """Assign 70/15/15 deterministically within 10 duration strata."""
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: (float(row["duration"]), row["sample_id"]))
    buckets = [ordered[i::10] for i in range(10)]
    for bucket in buckets:
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = round(n * 0.70)
        n_val = round(n * 0.15)
        for index, row in enumerate(bucket):
            row["split"] = "train" if index < n_train else "val" if index < n_train + n_val else "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data_root, output_root = args.data_root, args.output_root
    manifest_dir = output_root / "manifests"
    analysis_dir = output_root / "analysis"
    audio_dir = output_root / "audio_wav"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not audio_dir.exists() or not any(audio_dir.glob("*.wav")):
        with zipfile.ZipFile(data_root / "audio_wav.zip") as archive:
            archive.extractall(output_root)

    with (data_root / "CS-MSASR_fulldata.pkl").open("rb") as handle:
        data = pickle.load(handle)
    ids, texts = data["id"], data["raw_text"]
    if len(ids) != len(texts):
        raise RuntimeError(f"id/text length mismatch: {len(ids)} vs {len(texts)}")

    rows, issues = [], []
    observed_formats, wav_names = Counter(), set(path.name for path in audio_dir.glob("*.wav"))
    for sample_id, text in zip(ids, texts):
        filename = expected_wav(sample_id)
        path = audio_dir / filename
        normalized = normalize_text(text)
        issue = []
        if not path.exists():
            issue.append("missing_wav")
            duration = 0.0
        else:
            try:
                with wave.open(str(path), "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
                    observed_formats[(wav.getframerate(), wav.getnchannels(), wav.getsampwidth() * 8, wav.getcomptype())] += 1
            except Exception as error:
                issue.append(f"unreadable_wav:{type(error).__name__}")
                duration = 0.0
        if not str(text).strip(): issue.append("empty_reference")
        if not normalized: issue.append("empty_after_normalization")
        if "<" in str(text) or ">" in str(text): issue.append("possible_markup")
        if duration and len(normalized) / duration > 16: issue.append("high_char_rate")
        if duration and len(normalized) / duration < 0.2: issue.append("low_char_rate")
        row = {"sample_id": sample_id, "audio_path": str(path), "reference_text": str(text),
               "reference_normalized": normalized, "duration": round(duration, 6),
               "num_characters": len(normalized), "split": ""}
        rows.append(row)
        if issue:
            issues.append({**row, "issues": "|".join(issue)})

    if set(expected_wav(sample_id) for sample_id in ids) != wav_names:
        issues.append({"sample_id": "__dataset__", "issues": "id_wav_set_mismatch"})
    balanced_split(rows)
    fields = ["sample_id", "audio_path", "reference_text", "reference_normalized", "duration", "num_characters", "split"]
    for name, subset in [("manifest_all.csv", rows), ("train.csv", [r for r in rows if r["split"] == "train"]),
                         ("val.csv", [r for r in rows if r["split"] == "val"]), ("test.csv", [r for r in rows if r["split"] == "test"])]:
        with (manifest_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(subset)
    with (analysis_dir / "transcript_suspects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["issues"])
        writer.writeheader(); writer.writerows(issues)
    summary = {"n_samples": len(rows), "n_wavs": len(wav_names), "formats": {str(k): v for k, v in observed_formats.items()},
               "split_counts": Counter(r["split"] for r in rows), "split_hours": {split: round(sum(r["duration"] for r in rows if r["split"] == split) / 3600, 4) for split in ("train", "val", "test")},
               "duration_seconds": {"mean": statistics.mean(r["duration"] for r in rows), "median": statistics.median(r["duration"] for r in rows), "total": sum(r["duration"] for r in rows)},
               "rule": "sample-level, duration-stratified 70/15/15 split with seed 42", "initial_suspects": len(issues)}
    (analysis_dir / "data_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
