"""Run one FunASR-based Chinese ASR baseline on an existing CS-Mage manifest.

The script deliberately performs per-utterance inference: it gives a reliable
per-sample latency and lets an individual failure be recorded without losing
the remainder of the fixed test set.  It is intended for Paraformer and
SenseVoice direct-inference experiments only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from evaluate_asr import edit_counts
from normalize_text import normalize_text


TAG = re.compile(r"<\|[^|]*\|>|<[^>]+>")


def clean_sensevoice(text: str) -> str:
    """Discard SenseVoice language/emotion/event metadata but retain transcript."""
    return TAG.sub("", str(text)).strip()


def result_text(result: object) -> str:
    if isinstance(result, list) and result:
        result = result[0]
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-kind", choices=["paraformer", "sensevoice"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from funasr import AutoModel

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    if args.model_kind == "paraformer":
        model = AutoModel(model=args.model_id, vad_model="fsmn-vad", punc_model="ct-punc-c", device=args.device)
        generation_kwargs = {"language": "zh", "use_itn": False}
        name = "Paraformer-Large"
    else:
        model = AutoModel(
            model=args.model_id,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=args.device,
        )
        generation_kwargs = {"language": "zh", "use_itn": False}
        name = "SenseVoiceSmall"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "audio_path", "reference_text", "duration", "prediction_raw",
        "prediction", "prediction_normalized", "CER", "substitutions", "deletions",
        "insertions", "decode_seconds", "inference_time", "error",
    ]
    total_started = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            began = time.perf_counter()
            error = ""
            raw = ""
            try:
                generated = model.generate(input=row["audio_path"], **generation_kwargs)
                raw = result_text(generated).strip()
            except Exception as exc:  # preserve a row even for a failed utterance
                error = f"{type(exc).__name__}: {exc}"
            prediction = clean_sensevoice(raw) if args.model_kind == "sensevoice" else raw
            reference_normalized = normalize_text(row["reference_text"])
            prediction_normalized = normalize_text(prediction)
            counts = edit_counts(list(reference_normalized), list(prediction_normalized))
            elapsed = time.perf_counter() - began
            writer.writerow({
                "sample_id": row["sample_id"], "audio_path": row["audio_path"],
                "reference_text": row["reference_text"], "duration": row["duration"],
                "prediction_raw": raw, "prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "CER": counts["errors"] / max(1, len(reference_normalized)),
                "substitutions": counts["substitutions"], "deletions": counts["deletions"],
                "insertions": counts["insertions"], "decode_seconds": round(elapsed, 6),
                "inference_time": round(elapsed, 6), "error": error,
            })
            if index % 10 == 0 or index == len(rows):
                print(json.dumps({"processed": index, "total": len(rows), "elapsed_seconds": round(time.perf_counter() - total_started, 2)}, ensure_ascii=False), flush=True)

    elapsed = time.perf_counter() - total_started
    metadata = {
        "model_name": name, "model_source": "FunASR/ModelScope official model",
        "model_id": args.model_id, "setting": "Direct inference", "split": args.manifest.stem,
        "decoding_parameters": {"language": "zh", "use_itn": False, "per_utterance": True},
        "prediction_file": str(args.output), "inference_time": elapsed,
        "audio_seconds": sum(float(row["duration"]) for row in rows),
        "real_time_factor": elapsed / max(1e-9, sum(float(row["duration"]) for row in rows)),
        "random_seed": 42, "command": " ".join(sys.argv), "status": "inference_completed",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
