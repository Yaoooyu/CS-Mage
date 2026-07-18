# Fixed splits

All MSA experiments, including MAGE-Fusion, its ablations, and all baselines,
use the fixed sample-ID reconstructed five-fold protocol with seed 20260715.
Every outer fold is 759 train, 109 validation, and 217 test samples; the five
test partitions cover all 1,085 samples once.

All ASR experiments use the fixed duration-stratified sample-ID split with seed
42: 760 train, 160 validation, and 165 test samples.

The repository provides fixed split CSVs under `splits/msa/` and `splits/asr/`.
MSA rows also contain the published multimodal class label.

`splits/id_split_index.csv` consolidates all assignments by `sample_id`: five
MSA fold-role columns and one ASR split column. Its schema and usage are
documented in `ID_BASED_SPLIT_PROTOCOL.md`.

`CS-Mage_fulldata.pkl` contains all samples without an embedded
train/validation/test assignment. The split files in this repository are the
authoritative protocol for reproducing the reported experiments.
