# Fixed splits

MSA uses the formal sample-level reconstructed five-fold protocol, seed 20260715. Every outer fold is 759 train, 109 validation, and 217 test samples; the five test partitions cover all 1,085 samples once. It is not speaker-independent.

ASR uses a fixed duration-stratified sample-level split, seed 42: 760 train, 160 validation, 165 test samples. It is not speaker-independent.

The repository provides fixed split CSVs under `splits/msa/` and `splits/asr/`.
MSA rows also contain the published multimodal class label.

`CS-Mage_fulldata.pkl` contains all samples without an embedded
train/validation/test assignment. The split files in this repository are the
authoritative protocol for reproducing the reported experiments.
