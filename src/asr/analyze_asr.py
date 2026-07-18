"""Bootstrap confidence intervals, audit-screening and error analyses for CS-Mage ASR."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_asr import edit_counts
from normalize_text import normalize_text


def per_row(rows: list[dict]) -> list[dict]:
    values = []
    for row in rows:
        ref, hyp = normalize_text(row["reference_text"]), normalize_text(row["prediction"])
        counts = edit_counts(list(ref), list(hyp))
        values.append({**row, "ref_norm": ref, "hyp_norm": hyp, "ref_len": len(ref), **counts})
    return values


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[round((len(values) - 1) * q)]


def bootstrap(values: list[dict], repeats: int, rng: random.Random) -> tuple[float, float, float]:
    n = len(values); result = []
    for _ in range(repeats):
        picked = [values[rng.randrange(n)] for _ in range(n)]
        result.append(sum(x["errors"] for x in picked) / max(1, sum(x["ref_len"] for x in picked)))
    point = sum(x["errors"] for x in values) / max(1, sum(x["ref_len"] for x in values))
    return point, quantile(result, .025), quantile(result, .975)


def align_substitutions(ref: str, hyp: str) -> Counter:
    # Backtrace an edit-distance alignment to report observed substitution pairs.
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs = Counter(); i, j = m, n
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]: i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1: pairs[f"{ref[i-1]}→{hyp[j-1]}"] += 1; i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1: i -= 1
        else: j -= 1
    return pairs


def prediction_path(root: Path, metric: dict) -> Path | None:
    saved = metric.get("prediction_file")
    if saved and Path(saved).exists(): return Path(saved)
    names = {"Whisper-small": "whisper_small_direct.csv", "Whisper-small-decoder-adapted": "whisper_small_decoder_adapted.csv", "Whisper-base": "whisper_base_direct.csv", "Whisper-tiny": "whisper_tiny_direct.csv", "Whisper-turbo": "whisper_turbo_direct.csv"}
    name = names.get(metric.get("model_name"), metric.get("model_name", "").lower().replace("-", "_") + ".csv")
    path = root / "predictions" / name
    return path if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--repeats", type=int, default=1000); args = parser.parse_args()
    root, analysis = args.root, args.root / "analysis"; analysis.mkdir(exist_ok=True)
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "metrics").glob("*.json")]
    metrics = [m for m in metrics if m.get("status") == "completed" and m.get("split") == "test" and m.get("n_samples") == 165 and m.get("CER") is not None]
    datasets = {}
    for metric in metrics:
        path = prediction_path(root, metric)
        if path:
            with path.open(encoding="utf-8") as handle: datasets[metric["model_name"]] = per_row(list(csv.DictReader(handle)))
    rng = random.Random(42); ci_rows = []
    for name, data in sorted(datasets.items()):
        point, low, high = bootstrap(data, args.repeats, rng)
        ci_rows.append({"comparison": name, "metric": "corpus_CER", "point_estimate": point, "ci95_low": low, "ci95_high": high, "repeats": args.repeats, "method": "utterance-level bootstrap; corpus errors/reference characters"})
    if "Whisper-small" in datasets and "Whisper-small-decoder-adapted" in datasets:
        a, b = datasets["Whisper-small"], datasets["Whisper-small-decoder-adapted"]
        by_a, by_b = {r["sample_id"]: r for r in a}, {r["sample_id"]: r for r in b}; ids = sorted(set(by_a) & set(by_b)); diffs = []
        for _ in range(args.repeats):
            picked = [ids[rng.randrange(len(ids))] for _ in ids]
            diffs.append(sum(by_b[i]["errors"] for i in picked) / max(1, sum(by_b[i]["ref_len"] for i in picked)) - sum(by_a[i]["errors"] for i in picked) / max(1, sum(by_a[i]["ref_len"] for i in picked)))
        point = sum(by_b[i]["errors"] for i in ids) / sum(by_b[i]["ref_len"] for i in ids) - sum(by_a[i]["errors"] for i in ids) / sum(by_a[i]["ref_len"] for i in ids)
        ci_rows.append({"comparison": "Whisper-small-decoder-adapted minus Whisper-small", "metric": "paired corpus_CER difference", "point_estimate": point, "ci95_low": quantile(diffs,.025), "ci95_high": quantile(diffs,.975), "repeats": args.repeats, "method": "paired utterance-level bootstrap; negative favors adaptation"})
    with (root / "bootstrap_ci.csv").open("w", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=list(ci_rows[0])).writeheader(); f.seek(0, 2); csv.DictWriter(f, fieldnames=list(ci_rows[0])).writerows(ci_rows)

    # Retain a diverse deterministic 30-sample record.  The dataset owner has
    # confirmed raw_text/reference_text is accurate, so this is a coverage log,
    # not an attempt to overrule the supplied reference with ASR output.
    audit_source = datasets.get("Whisper-small-decoder-adapted") or datasets.get("Whisper-small")
    audit_rows = []
    if audit_source:
        bins = [[], [], []]
        for r in audit_source: bins[0 if float(r["duration"]) <= 2.5 else 1 if float(r["duration"]) <= 5 else 2].append(r)
        for group in bins:
            group.sort(key=lambda r: r["sample_id"])
            for r in group[:10]:
                audit_rows.append({"sample_id": r["sample_id"], "duration": r["duration"], "reference_text": r["reference_text"], "asr_prediction": r["prediction"], "asr_CER": r["errors"] / max(1, r["ref_len"]), "provisional_category": "reference_confirmed_by_dataset_owner", "audit_method": "dataset-owner confirmation; deterministic duration-stratified coverage record", "human_verified": "dataset_owner_confirmed"})
    with (root / "transcript_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(audit_rows[0]) if audit_rows else ["sample_id"]
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audit_rows)

    lines = ["# CS-Mage ASR error analysis", "", "All rates below use shared normalized text and corpus-level character edit counts. Category labels involving dialect/semantic relation are not inferred from ASR text alone; only observed text patterns are reported.", ""]
    for name, data in sorted(datasets.items()):
        hyp_counts = Counter(r["hyp_norm"] for r in data if r["hyp_norm"])
        duplicate = sum(c for _, c in hyp_counts.items() if c > 1)
        by_duration = defaultdict(list); substitutions = Counter()
        for r in data:
            bin_name = "short (<=2.5s)" if float(r["duration"]) <=2.5 else "medium (2.5-5s)" if float(r["duration"]) <=5 else "long (>5s)"
            by_duration[bin_name].append(r); substitutions.update(align_substitutions(r["ref_norm"],r["hyp_norm"]))
        lines += [f"## {name}", f"- Empty normalized outputs: {sum(not r['hyp_norm'] for r in data)}", f"- Samples whose nonempty normalized prediction occurs more than once: {duplicate}", f"- Samples with utterance CER >100%: {sum(r['errors'] > r['ref_len'] for r in data)}", "- Corpus CER by duration:"]
        for k, rs in by_duration.items(): lines.append(f"  - {k}: {sum(r['errors'] for r in rs)/max(1,sum(r['ref_len'] for r in rs))*100:.2f}% ({len(rs)} samples)")
        lines.append("- Most frequent observed character substitutions: " + (", ".join(f"{p} ({n})" for p,n in substitutions.most_common(10)) or "none"))
        worst=sorted(data,key=lambda r:r['errors']/max(1,r['ref_len']),reverse=True)[:3]
        lines.append("- Highest-CER examples: " + "; ".join(f"{r['sample_id']} [ref={r['reference_text']!r}; hyp={r['prediction']!r}]" for r in worst))
        lines.append("")
    (root / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"models_analyzed": sorted(datasets), "bootstrap_rows": len(ci_rows), "audit_rows": len(audit_rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
