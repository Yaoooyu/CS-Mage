# Release inventory

- `src/msa/`: actual formal-v2 and dependency modules, retained in their executed layout.
- `src/training/`: CS-Mage wrappers for third-party and reproduced baselines.
- `src/asr/`: copied ASR preparation, normalization, evaluation, inference/adaptation utilities; completeness is tracked in unresolved items.
- `assets/`: tokenizer configuration/vocabulary only; no pretrained weights.
- `configs/`: formal-v2, ablation, and ASR records.
- `splits/`: required sanitized split metadata is not yet populated.

Excluded by design: raw videos/audio/features, pkl files, checkpoints, optimizer state, predictions, logs, caches, model weights, Git metadata, and IDE settings.
