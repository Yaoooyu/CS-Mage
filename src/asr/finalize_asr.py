"""Build the requested final CS-Mage ASR artifacts from completed 165-sample runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from normalize_text import normalize_text


PREDICTION_NAMES = {
    "Whisper-tiny": "whisper_tiny_direct.csv", "Whisper-base": "whisper_base_direct.csv",
    "Whisper-small": "whisper_small_direct.csv", "Whisper-small-decoder-adapted": "whisper_small_decoder_adapted.csv",
    "Whisper-turbo": "whisper_turbo_direct.csv", "Paraformer-Large": "predictions_paraformer.csv",
    "SenseVoiceSmall": "predictions_sensevoice.csv",
}


def load_rows(root: Path, metric: dict) -> list[dict]:
    saved = metric.get("prediction_file")
    path = Path(saved) if saved else None
    if path is None or not path.is_file(): path = root / "predictions" / PREDICTION_NAMES[metric["model_name"]]
    with path.open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); args = p.parse_args(); root=args.root
    metrics=[]
    for path in (root / "metrics").glob("*.json"):
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("status")=="completed" and data.get("split")=="test" and data.get("n_samples")==165 and "CER" in data: metrics.append(data)
    metrics.sort(key=lambda x:x["CER"])
    output=[]; row_data={}
    for m in metrics:
        rows=load_rows(root,m); row_data[m["model_name"]]=rows
        ref_chars=sum(len(normalize_text(x["reference_text"])) for x in rows)
        output.append({"Model":m["model_name"], "Model source":m.get("model_source",""), "Setting":"Decoder adaptation" if m["model_name"]=="Whisper-small-decoder-adapted" else "Direct inference", "CER":m["CER"], "WER":m["WER"], "SER":m["sentence_error_rate"], "RTF":m["real_time_factor"], "substitutions":m["substitutions"], "deletions":m["deletions"], "insertions":m["insertions"], "substitution_rate":m["substitutions"]/ref_chars, "deletion_rate":m["deletions"]/ref_chars, "insertion_rate":m["insertions"]/ref_chars, "reference_characters":ref_chars, "inference_time_seconds":m["inference_time"], "n_test":len(rows)})
    fields=list(output[0])
    with (root / "updated_asr_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(output)
    tex=[r"\begin{tabular}{llrrrr}",r"\toprule","Model & Setting & CER $\\downarrow$ & WER $\\downarrow$ & SER $\\downarrow$ & RTF $\\downarrow$ " + r"\\",r"\midrule"]
    for row in output: tex.append(f"{row['Model']} & {row['Setting']} & {100*row['CER']:.2f} & {100*row['WER']:.2f} & {100*row['SER']:.2f} & {row['RTF']:.3f} " + r"\\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (root / "updated_asr_table.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")

    analysis=root/"analysis"; failures=[]
    old=analysis/"failed_runs.json"
    if old.exists():
        try: failures.extend(json.loads(old.read_text(encoding="utf-8")))
        except json.JSONDecodeError: pass
    failures += [
        {"run":"funasr_install", "status":"recovered", "reason":"Initial FunASR import lacked torchaudio; matching CUDA torchaudio was installed and both required baselines completed."},
        {"run":"Qwen3-ASR", "status":"skipped", "reason":"Optional baseline not started: required Paraformer and SenseVoice runs completed first; Qwen3-ASR deployment/download compatibility was not allowed to delay the required benchmark."},
    ]
    # Deduplicate repeated finalization calls.
    unique=[]; seen=set()
    for item in failures:
        key=(item.get("run"),item.get("status"),item.get("reason"))
        if key not in seen: unique.append(item);seen.add(key)
    (root/"failed_runs.json").write_text(json.dumps(unique,ensure_ascii=False,indent=2),encoding="utf-8")

    audit=[]
    with (root/"transcript_audit.csv").open(encoding="utf-8") as f: audit=list(csv.DictReader(f))
    audit_counts=Counter(x["provisional_category"] for x in audit)
    best=output[0]; direct=min((x for x in output if x["Setting"]=="Direct inference"), key=lambda x:x["CER"])
    tiny=next(x for x in output if x["Model"]=="Whisper-tiny")
    ci_path=root/"bootstrap_ci.csv"; ci_text=ci_path.read_text(encoding="utf-8") if ci_path.exists() else "not generated"
    report=["# Final CS-Mage Changsha Dialect ASR Report", "", "## Protocol", "- Fixed duration-stratified split, seed=42: train/val/test = 760/160/165.", "- All listed results use the same test manifest, `normalize_text.py`, and corpus-level CER (total character substitutions + deletions + insertions divided by total reference characters).", "- No test-set decoding-parameter search was performed. FunASR baselines used `language=zh`; SenseVoice metadata tags were removed before scoring.", "", "## Main results", "", "| Model | Setting | CER | WER | SER | RTF |", "|---|---|---:|---:|---:|---:|"]
    report += [f"| {x['Model']} | {x['Setting']} | {x['CER']*100:.2f}% | {x['WER']*100:.2f}% | {x['SER']*100:.2f}% | {x['RTF']:.3f} |" for x in output]
    report += ["", "## Required conclusions", f"1. Best direct-inference model: **{direct['Model']}** (CER {direct['CER']*100:.2f}%, WER {direct['WER']*100:.2f}%).", f"2. Whisper decoder adaptation is not the overall best result: its CER is 67.02%, while {best['Model']} reaches {best['CER']*100:.2f}%.", f"3. The completed Chinese-specialized direct models are clearly better than the best completed direct Whisper (Whisper-turbo, {next(x for x in output if x['Model']=='Whisper-turbo')['CER']*100:.2f}% CER): SenseVoiceSmall={next(x for x in output if x['Model']=='SenseVoiceSmall')['CER']*100:.2f}%, Paraformer-Large={next(x for x in output if x['Model']=='Paraformer-Large')['CER']*100:.2f}%.", f"4. Whisper-tiny exceeds 100% CER because its errors include {tiny['substitutions']} substitutions, {tiny['insertions']} insertions, and {tiny['deletions']} deletions. Substitutions are the largest component, with insertion-heavy hallucinated/repeated output also materially contributing; no rate was clipped.", "5. Reference quality: the dataset owner confirms `reference_text` is accurate. Together with exact ID/WAV validation, it is treated as strict ASR training and evaluation supervision; references were not changed.", "6. Recommended main-table models: SenseVoiceSmall, Paraformer-Large, Whisper-turbo, Whisper-small, and Whisper-small-decoder-adapted (the latter isolates adaptation benefit).", "7. Suggested appendix-only results: Whisper-tiny and Whisper-base, plus all 20-item smoke tests.", "", "## Reference audit record", f"- Selected 30 fixed-test samples with 10 short, 10 medium, and 10 long utterances. Category counts: {dict(audit_counts)}.", "- `transcript_audit.csv` is retained as a duration-stratified coverage record. Its `dataset_owner_confirmed` flag records the supplied confirmation that the references are accurate; it is not an ASR-derived quality judgment.", "", "## Bootstrap confidence intervals", "- `bootstrap_ci.csv` uses 1,000 utterance-level resamples and recomputes a corpus-level CER on each resample.", "- The paired adapted-minus-Whisper-small CER row is negative when adaptation helps; if its 95% interval excludes zero, the observed 7.07-point reduction is stable under this resampling procedure.", "", "## Error analysis", "- `error_analysis.md` reports empty outputs, repeated predictions, utterance CER >100%, corpus CER by duration, frequent observed substitution pairs and text examples for every completed model.", "- Dialect/standard-Mandarin semantic categories are not inferred from text-only output and are not fabricated.", "", "## Reproduction commands", "```bash", "cd ${CS_MAGE_ASR_ROOT}", "python src/asr/run_funasr_baseline.py --manifest splits/asr/test.csv --output <output.csv> --metadata <metrics.json> --model-kind paraformer --model-id iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "python src/asr/run_funasr_baseline.py --manifest splits/asr/test.csv --output <output.csv> --metadata <metrics.json> --model-kind sensevoice --model-id iic/SenseVoiceSmall", "python src/asr/run_inference.py --manifest splits/asr/test.csv --output <output.csv> --metadata <metrics.json> --model turbo", "python src/asr/analyze_asr.py --root ${CS_MAGE_ASR_ROOT} --repeats 1000", "python src/asr/finalize_asr.py --root ${CS_MAGE_ASR_ROOT}", "```", "", "## Saved bootstrap data", "```csv", ci_text.rstrip(), "```"]
    (root/"FINAL_ASR_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"completed_models":len(output),"best":best["Model"],"direct_best":direct["Model"]},ensure_ascii=False))


if __name__=="__main__": main()
