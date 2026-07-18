"""Run a Whisper direct-inference ASR baseline and save all utterance predictions."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=["tiny", "base", "small", "turbo"]); parser.add_argument("--checkpoint", type=Path); parser.add_argument("--training-summary", type=Path); parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=int, default=1); parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    # imageio-ffmpeg provides a user-space binary on hosts without system ffmpeg.
    try:
        import imageio_ffmpeg
        ffmpeg_dir = Path(tempfile.gettempdir()) / "cs_mage_ffmpeg"
        ffmpeg_dir.mkdir(exist_ok=True)
        ffmpeg_link = ffmpeg_dir / "ffmpeg"
        if not ffmpeg_link.exists(): ffmpeg_link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    import torch
    import whisper
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.limit: rows = rows[:args.limit]
    model = whisper.load_model(args.model, device=args.device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model_state"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "audio_path", "reference_text", "duration", "prediction", "decode_seconds", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(rows, 1):
            began = time.perf_counter(); error = ""
            try:
                result = model.transcribe(row["audio_path"], language="zh", task="transcribe", fp16=args.device == "cuda", beam_size=args.beam_size, temperature=0, verbose=False, condition_on_previous_text=False)
                prediction = result.get("text", "").strip()
            except Exception as exc:
                prediction, error = "", f"{type(exc).__name__}: {exc}"
            record = {field: row.get(field, "") for field in fields}
            record.update({"prediction": prediction, "decode_seconds": round(time.perf_counter() - began, 6), "error": error})
            writer.writerow(record)
            if index % 10 == 0 or index == len(rows): print(json.dumps({"processed": index, "total": len(rows), "elapsed_seconds": round(time.perf_counter() - started, 2)}, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started
    training_summary = json.loads(args.training_summary.read_text(encoding="utf-8")) if args.training_summary else {}
    metadata = {"model_name": f"Whisper-{args.model}" + ("-decoder-adapted" if args.checkpoint else ""), "model_source": "openai-whisper 20240930", "setting": "Fine-tuned decoder" if args.checkpoint else "Direct inference", "split": args.manifest.stem,
                "decoding_parameters": {"language": "zh", "task": "transcribe", "beam_size": args.beam_size, "temperature": 0, "condition_on_previous_text": False},
                "inference_time": elapsed, "audio_seconds": sum(float(row["duration"]) for row in rows), "real_time_factor": elapsed / max(1e-9, sum(float(row["duration"]) for row in rows)),
                "training_time": training_summary.get("training_time", 0), "best_validation_CER": training_summary.get("best_validation_CER"), "random_seed": 42, "command": " ".join(__import__("sys").argv), "status": "inference_completed"}
    args.metadata.parent.mkdir(parents=True, exist_ok=True); args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__": main()
