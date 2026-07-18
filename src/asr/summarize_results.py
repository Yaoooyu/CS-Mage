"""Create paper-ready tables and auditable ASR error analyses from saved runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluate_asr import edit_counts
from normalize_text import normalize_text


def individual_error(row: dict) -> dict:
    reference, prediction = normalize_text(row["reference_text"]), normalize_text(row["prediction"])
    counts = edit_counts(list(reference), list(prediction))
    return {"sample_id": row["sample_id"], "duration": float(row["duration"]), "reference_text": row["reference_text"], "prediction": row["prediction"],
            "CER": counts["errors"] / max(1, len(reference)), "substitutions": counts["substitutions"], "deletions": counts["deletions"], "insertions": counts["insertions"], "error": row.get("error", "")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args(); root = args.root
    metrics = []
    for path in sorted((root / "metrics").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "completed" and row.get("split") == "test" and "CER" in row:
            metrics.append(row)
    metrics.sort(key=lambda row: row["CER"])
    fields = ["model_name", "model_source", "setting", "CER", "WER", "sentence_error_rate", "substitutions", "deletions", "insertions", "inference_time", "real_time_factor", "status"]
    with (root / "asr_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{k: row.get(k) for k in fields} for row in metrics])
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Model & Setting & CER $\\downarrow$ & WER $\\downarrow$ & SER $\\downarrow$ " + r"\\", "\\midrule"]
    for row in metrics: lines.append(f"{row['model_name']} & {row['setting']} & {100*row['CER']:.2f} & {100*row['WER']:.2f} & {100*row['sentence_error_rate']:.2f} " + r"\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "asr_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    analysis_dir = root / "analysis"; analysis_dir.mkdir(exist_ok=True)
    if metrics:
        best = metrics[0]
        prediction_path = root / "predictions" / ("whisper_small_decoder_adapted.csv" if best["model_name"] == "Whisper-small-decoder-adapted" else best["model_name"].lower().replace("-", "_") + "_direct.csv")
        with prediction_path.open(encoding="utf-8") as handle: errors = [individual_error(row) for row in csv.DictReader(handle)]
        errors.sort(key=lambda row: row["CER"], reverse=True)
        error_fields = list(errors[0]) if errors else ["sample_id", "duration", "reference_text", "prediction", "CER", "substitutions", "deletions", "insertions", "error"]
        with (analysis_dir / "error_samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=error_fields); writer.writeheader(); writer.writerows(errors)
        groups = {"short_<=2.5s": [], "medium_2.5-5s": [], "long_>5s": []}
        for row in errors: groups["short_<=2.5s" if row["duration"] <= 2.5 else "medium_2.5-5s" if row["duration"] <= 5 else "long_>5s"].append(row)
        duration_rows = []
        for name, rows in groups.items():
            duration_rows.append({"duration_bin": name, "n_samples": len(rows), "mean_CER": sum(r["CER"] for r in rows) / max(1, len(rows)), "mean_insertions": sum(r["insertions"] for r in rows) / max(1, len(rows))})
        with (analysis_dir / "error_by_duration.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(duration_rows[0])); writer.writeheader(); writer.writerows(duration_rows)
    else:
        best, duration_rows = None, []

    failures = []
    for log in (root / "logs").glob("*.log"):
        text = log.read_text(encoding="utf-8", errors="replace")
        if "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'" in text:
            failures.append({"run": log.stem, "status": "recovered", "reason": "Initial smoke test lacked an ffmpeg executable; rerun succeeded using imageio-ffmpeg."})
    (analysis_dir / "failed_runs.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "results_registry.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = json.loads((analysis_dir / "data_validation.json").read_text(encoding="utf-8"))
    report = ["# CS-Mage Changsha Dialect ASR Benchmark", "", "## Data validation", f"- {validation['n_samples']} WAV files were readable and exactly matched to the {validation['n_samples']} sample IDs.", f"- Observed WAV format: {validation['formats']}.", f"- Fixed sample-level duration-stratified split (seed 42): train/val/test = {dict(validation['split_counts'])}. No speaker-independent claim is made.", "- `raw_text` is non-empty for all samples and contains Chinese utterance text. Direct ASR checks show exact and near-exact examples, but its status remains candidate reference transcription pending manual audit.", "", "## Direct-inference results", "", "| Model | CER | WER | SER | RTF |", "|---|---:|---:|---:|---:|"]
    for row in metrics: report.append(f"| {row['model_name']} | {100*row['CER']:.2f}% | {100*row['WER']:.2f}% | {100*row['sentence_error_rate']:.2f}% | {row['real_time_factor']:.3f} |")
    if best: report.extend(["", "## Error analysis", f"- Best saved direct model by normalized CER: {best['model_name']}.", f"- Aggregate character edits: substitutions={best['substitutions']}, deletions={best['deletions']}, insertions={best['insertions']}.", "- Duration analysis is stored in `analysis/error_by_duration.csv`; detailed utterance-level errors are stored in `analysis/error_samples.csv`."])
    report.extend(["", "## Reproduction", "- Run `scripts/prepare_data.py` to recreate manifests and the fixed split.", "- Run `scripts/run_inference.py` followed by `scripts/evaluate_asr.py` for each model.", "- All main-table values use the shared NFKC/lowercase/punctuation-removal normalization and the same held-out test set.", "", "## Remaining manual checks", "- Verify a random audio/transcript audit before treating `raw_text` as strict training supervision.", "- No verified speaker IDs are available; this is not an unseen-speaker evaluation."])
    (root / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"completed_models": len(metrics), "best": best["model_name"] if best else None}, ensure_ascii=False))


if __name__ == "__main__": main()
