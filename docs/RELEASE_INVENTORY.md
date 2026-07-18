# Release inventory

- `src/msa/`: MAGE-Fusion and dependency modules, retained in their executed layout.
- `src/training/`: CS-Mage wrappers for third-party and reproduced baselines.
- `src/asr/`: ASR preparation, normalization, evaluation, inference, adaptation,
  and reporting utilities.
- `assets/`: tokenizer configuration/vocabulary only; no pretrained weights.
- `configs/`: MAGE-Fusion, ablation, and ASR records.
- `splits/`: fixed sanitized MSA five-fold and ASR train/validation/test split
  metadata.

Dataset files, including `CS-Mage_fulldata.pkl`, are distributed through the
dataset access links documented in `DATA_ACCESS_AND_ETHICS.md` rather than
stored in this code repository.

The repository excludes checkpoints, optimizer state, predictions, logs,
caches, model weights, Git metadata, and IDE settings.
