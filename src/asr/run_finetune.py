"""Low-resource Whisper decoder adaptation with smoke mode and CER early stopping."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from evaluate_asr import edit_counts
from normalize_text import normalize_text


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(normalize_text(reference)), list(normalize_text(hypothesis))
    return edit_counts(ref, hyp)["errors"] / max(1, len(ref))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True); parser.add_argument("--val", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["tiny", "base", "small"], default="small"); parser.add_argument("--epochs", type=int, default=6); parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4); parser.add_argument("--lr", type=float, default=1e-5); parser.add_argument("--patience", type=int, default=2); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
        ffmpeg_dir = Path(tempfile.gettempdir()) / "cs_mage_ffmpeg"; ffmpeg_dir.mkdir(exist_ok=True)
        ffmpeg_link = ffmpeg_dir / "ffmpeg"
        if not ffmpeg_link.exists(): ffmpeg_link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    import torch
    import torch.nn.functional as F
    import whisper
    from whisper.tokenizer import get_tokenizer
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows = list(csv.DictReader(args.train.open(encoding="utf-8"))); val_rows = list(csv.DictReader(args.val.open(encoding="utf-8")))
    if args.smoke: train_rows, val_rows, args.epochs = train_rows[:32], val_rows[:20], 1
    model = whisper.load_model(args.model, device=device)
    for parameter in model.parameters(): parameter.requires_grad = False
    for parameter in model.decoder.parameters(): parameter.requires_grad = True
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def batch_loss(batch: list[dict]) -> torch.Tensor:
        mels, token_rows = [], []
        for row in batch:
            audio = whisper.pad_or_trim(whisper.load_audio(row["audio_path"]))
            mel = whisper.log_mel_spectrogram(audio).to(device)
            mels.append(mel)
            token_rows.append(prefix + tokenizer.encode(row["reference_text"]) + [tokenizer.eot])
        max_len = min(448, max(len(tokens) for tokens in token_rows))
        inputs = torch.full((len(batch), max_len - 1), tokenizer.eot, device=device, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        for i, tokens in enumerate(token_rows):
            tokens = tokens[:max_len]
            inputs[i, :len(tokens)-1] = torch.tensor(tokens[:-1], device=device)
            labels[i, :len(tokens)-1] = torch.tensor(tokens[1:], device=device)
        mel_batch = torch.stack(mels)
        # Weak SpecAugment is applied only to the training mel features.
        if model.training:
            for mel in mel_batch:
                start = random.randrange(0, 2800); mel[:, start:start + 80] = 0
        logits = model(mel_batch, inputs)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def validate() -> float:
        model.eval(); values = []
        for row in val_rows:
            result = model.transcribe(row["audio_path"], language="zh", task="transcribe", fp16=device == "cuda", beam_size=1, temperature=0, verbose=False, condition_on_previous_text=False)
            values.append(cer(row["reference_text"], result.get("text", "")))
        return float(np.mean(values))

    log, best_cer, stale, best_path, started = [], float("inf"), 0, args.output / "best.pt", time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_rows); optimizer.zero_grad(set_to_none=True); losses = []
        for begin in range(0, len(train_rows), args.batch_size):
            batch = train_rows[begin:begin + args.batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = batch_loss(batch) / args.accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.item() * args.accumulation))
            if ((begin // args.batch_size) + 1) % args.accumulation == 0 or begin + args.batch_size >= len(train_rows):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        validation_cer = validate(); record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_CER": validation_cer, "elapsed_seconds": time.time() - started}; log.append(record); print(json.dumps(record), flush=True)
        if validation_cer < best_cer:
            best_cer, stale = validation_cer, 0; torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_CER": best_cer, "args": vars(args)}, best_path)
        else:
            stale += 1
        if stale >= args.patience: break
    (args.output / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"best_validation_CER": best_cer, "best_checkpoint": str(best_path), "epochs_ran": len(log), "training_time": time.time() - started, "model": args.model, "smoke": args.smoke, "status": "completed"}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
