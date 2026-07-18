# CS-Mage release candidate

This directory is a **pre-publication release candidate** assembled from the code that was used for the CS-Mage experiments. It contains code, fixed split metadata, configurations, and release documentation; it intentionally excludes all raw data, features, checkpoints, model weights, caches, and experiment logs.

The primary MSA method is formal-v2 MAGE-Fusion. The implementation is kept in `src/msa/` in its executed module layout to preserve imports. Run it from that directory after providing the data root; do not treat this candidate as a one-command training package.

The fixed protocol is sample-level five-fold evaluation (759/109/217 train/validation/test per fold; seed 20260715). It is not a speaker-independent protocol. ASR uses a separate 760/160/165 sample-level duration-stratified split (seed 42).

See `docs/` and `reports/` before publication. In particular, `reports/UNRESOLVED_ITEMS.md` records materials that must be resolved before a public GitHub release.
