# CS-Mage

## A Multimodal Benchmark for Changsha Dialect Sentiment Analysis and Speech Recognition

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ffb7f87-aa79-453e-926c-1cfaab9d5e9f" alt="CS-MSASR Dataset Preview" width="400"/>
</p>

CS-Mage is a benchmark for Changsha-dialect multimodal sentiment analysis (MSA)
and automatic speech recognition (ASR). This repository provides the code,
fixed split metadata, configuration records, and documentation for the
CS-Mage protocols.

## Overview

CS-Mage contains 1,085 short Changsha-dialect clips. Each MSA sample has text,
audio, and visual feature sequences, a five-level multimodal sentiment label,
and three modality-specific sentiment labels. The benchmark supports:

| Task | Input | Output | Evaluation protocol |
|---|---|---|---|
| Multimodal sentiment analysis | BERT-base-Chinese text, MFCC audio, OpenFace visual features | Five-level classification and continuous sentiment prediction | Fixed five-fold evaluation |
| Changsha dialect ASR | Raw 16-kHz, 16-bit mono waveform | Normalized transcription | Fixed duration-stratified split |

| Modality | Representation | Dimension | Maximum padded length |
|---|---|---:|---:|
| Text | BERT-base-Chinese token representation | 768 | 39 tokens |
| Audio | MFCC sequence | 13 | 553 frames |
| Visual | OpenFace frame representation | 347 | 321 frames |

MFCC features are used only for MSA. ASR always consumes raw waveforms.

## Repository layout

```text
CS-Mage/
├── configs/                  # MAGE-Fusion, ablation, and ASR configuration records
├── splits/                   # Sanitized fixed MSA and ASR split metadata
├── src/
│   ├── msa/                  # MAGE-Fusion modules and MSA wrappers
│   ├── training/             # CS-Mage adapters for additional baselines
│   ├── asr/                  # ASR preparation, inference, adaptation, evaluation
│   └── data/                 # Fold construction and BERT-input utilities
├── assets/bert-base-chinese/ # Tokenizer and configuration assets
├── baselines/                # Baseline release notes
├── docs/                     # Protocol, format, preprocessing, attribution documents
├── requirements.txt
├── environment.yml
```

### Included

- The MAGE-Fusion implementation and ablation runner.
- MSA and ASR data interfaces, normalization, evaluation, and utility code.
- Fixed MSA five-fold and ASR train/validation/test split files.
- MAGE-Fusion, ablation, and ASR configuration records.
- CS-Mage-specific baseline wrappers, source notes, and citations.

### Dataset access

- **Google Drive:** https://drive.google.com/drive/folders/1g5zbyc6ZMVdqC95yfTl4lZZSIkK9V_E5?usp=drive_link
- **Baidu Netdisk:** https://pan.baidu.com/s/1lYznkyVZ0GsaDKosHb9fKQ
  (access code: `2cbi`)

The feature package is provided as `CS-Mage_fulldata.pkl`. It contains the
full collection and does not embed a train/validation/test assignment. For
reproducibility, use the fixed split metadata included in this repository:
the MSA five-fold protocol in `splits/msa/` and the ASR 760/160/165 split in
`splits/asr/`. Do not replace these official protocols with a newly sampled
8:1:1 split.

Dataset use is restricted to non-commercial scientific research and education.
See [docs/DATA_ACCESS_AND_ETHICS.md](docs/DATA_ACCESS_AND_ETHICS.md) for the
data-use conditions and ethical safeguards.

## Fixed evaluation protocols

### Multimodal sentiment analysis

- Fixed stratified five-fold evaluation used by MAGE-Fusion and all MSA
  baselines.
- 1,085 samples total; every fold has 759 train / 109 validation / 217 test.
- Random seed: `20260715`.
- Validation data select the model; test data are used only for final evaluation.
- Metrics: Acc2, Acc3, Acc5, weighted F1, MAE, and Pearson correlation.

The five files in `splits/msa/` contain the sanitized protocol. Each sample
appears in a test partition exactly once. The public CSV names are one-based
(`fold_1.csv` to `fold_5.csv`); the retained legacy MSA runner uses its original
zero-based `--fold` index.

### Speech recognition

- Fixed duration-stratified train/validation/test split used by every ASR
  experiment.
- 760 train / 160 validation / 165 test clips; seed `42`.
- Shared normalization: Unicode NFKC, ASCII lowercase, retention of CJK,
  `0-9`, and `a-z`, then removal of punctuation, whitespace, emoji, and tags.
- Metrics: corpus-level CER, Jieba-based WER, SER, RTF, and edit-error counts.

Use the ASR split CSVs to construct manifests and evaluate the fixed test set.

## MAGE-Fusion

The primary MSA method is **MAGE-Fusion**. It combines
modality-specific encoders, modality-target and pairwise agreement learning,
agreement-conditioned attention, aggregation/interaction/disagreement experts,
an evidence gate, unimodal auxiliary classifiers, and synchronous Mixup.

| Component | Setting |
|---|---|
| Hidden dimension / prediction head | 192 / 192 → 256 → 128 |
| Encoder / evidence / head dropout | 0.20 / 0.10 / 0.25 |
| Optimizer | AdamW; learning rate `6e-4`; weight decay `3e-4` |
| Training | Full batch; maximum 180 epochs; patience 30; clipping 4.0 |
| Validation score | `Corr - MAE + 0.35 * Acc2` |
| Agreement learning | `tau_t=1.0`; `lambda_ta=lambda_pa=0.01` |
| Entropy regularization | Modality `0.005`; expert `0.003` |
| Mixup | `lambda_0 ~ Beta(0.35, 0.35)`; `lambda=max(lambda_0, 0.5)` |

This is **truncated Mixup**, not reflection Mixup. Do not replace it with
`max(lambda_0, 1-lambda_0)`.

Use `src/msa/run_mage_fusion.py` to run MAGE-Fusion and its ablations. The
model module is `src/msa/run_mage_fusion_fold.py`; the authoritative settings
are in [configs/msa/mage_fusion.yaml](configs/msa/mage_fusion.yaml).

MAGE-Fusion ablation records are provided for only `L_ta`, without agreement bias,
without disagreement expert, without Mixup, and without annotation guidance.
The historical `L_ta + L_pa` without agreement-bias label and the current
without-agreement-bias label are the same computation graph; do not count them
as separate variants.

## Baselines

The 18 documented MSA baselines are `LMF`, `MLMF`, `TFN`, `MTFN`, `LF-DNN`,
`MLF-DNN`, `EF-LSTM`, `MFN`, `Graph-MFN`, `MFM`, `MulT`, `Self-MM`, `MMIM`,
`CLGSI`, `KuDA`, `CubeMLP`, `ConFEDE`, and `EUAR`.

All were evaluated under the same reconstructed five-fold protocol and unified
metric implementation. Official implementations were reused whenever available;
unavailable methods were reproduced from their published descriptions. CubeMLP
and EUAR are paper-faithful reproductions, not official-code runs.

This repository includes CS-Mage adapters, not unverified third-party source
redistributions. See [docs/BASELINE_SOURCES.md](docs/BASELINE_SOURCES.md) for
the implementation status and [docs/BASELINE_REFERENCES.md](docs/BASELINE_REFERENCES.md)
for citations and upstream links.

## ASR utilities

| File | Purpose |
|---|---|
| `prepare_data.py` | Builds and validates ASR manifests from a local data root |
| `normalize_text.py` | Shared reference/prediction normalization |
| `evaluate_asr.py` | Corpus-level CER, WER, SER, RTF, and edit-error evaluation |
| `run_inference.py` | Whisper direct inference |
| `run_finetune.py` | Whisper decoder-only adaptation |
| `run_funasr_baseline.py` | Paraformer and SenseVoice direct inference |
| `analyze_asr.py`, `summarize_results.py`, `finalize_asr.py` | Analysis and reporting utilities |

SenseVoice language, emotion, and acoustic-event tags are removed before shared
normalization and evaluation. Paraformer and SenseVoice model IDs are supplied
at execution time.

## Environment setup

The recorded environment uses Python 3.12.3, PyTorch 2.6.0+cu124, CUDA 12.4,
NumPy 2.1.3, and scikit-learn 1.9.0. ASR utilities additionally use FunASR
1.3.16, ModelScope 1.38.1, openai-whisper 20240930, and Jieba 0.42.1.

```bash
conda env create -f environment.yml
conda activate cs-mage
```

Alternatively, install a PyTorch build compatible with your CUDA driver and run:

```bash
pip install -r requirements.txt
```

## Running the retained utilities

Run each experiment with its corresponding script and explicitly provide the
local data and output paths. The MSA runner operates one fold at a time:

```bash
python src/asr/prepare_data.py --help
python src/asr/run_inference.py --help
python src/asr/run_funasr_baseline.py --help
python src/asr/run_finetune.py --help
```

For MSA, point `--root` to a local directory whose `data/` subdirectory contains
the original per-fold feature PKLs. For ASR, build local manifests from WAV
files, retain the fixed IDs in `splits/asr/`, and use the shared
`normalize_text.py` and `evaluate_asr.py` for every model.

Configuration files use placeholders such as `${CS_MAGE_DATA_ROOT}` and
`${CS_MAGE_OUTPUT_ROOT}`. Do not add local paths, credentials, checkpoints, or
raw data to tracked files.

## Documentation

| Document | Contents |
|---|---|
| [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) | Expected MSA PKL and ASR manifest interfaces |
| [docs/DATA_ACCESS_AND_ETHICS.md](docs/DATA_ACCESS_AND_ETHICS.md) | Dataset download links, permitted use, and ethics safeguards |
| [docs/SPLITS.md](docs/SPLITS.md) | Fixed MSA and ASR split protocol |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | MAGE-Fusion, ablation, and ASR configurations |
| [docs/PREPROCESSING.md](docs/PREPROCESSING.md) | MSA feature and ASR waveform preprocessing |
| [docs/BASELINE_SOURCES.md](docs/BASELINE_SOURCES.md) | Baseline implementation and release status |
| [docs/BASELINE_REFERENCES.md](docs/BASELINE_REFERENCES.md) | Baseline citations and upstream links |
| [docs/RELEASE_INVENTORY.md](docs/RELEASE_INVENTORY.md) | Repository components |

## Citation

Citation information will be added after publication.
