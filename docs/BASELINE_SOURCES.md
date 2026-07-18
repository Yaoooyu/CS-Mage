# Baseline sources and status

See `BASELINE_REFERENCES.md` for per-method citation keys, paper links, and attribution details.

All baselines were evaluated under the same reconstructed five-fold protocol and unified metric implementation. Official implementations were reused whenever available, while unavailable methods were reproduced following their published descriptions.

| Methods | Status | CS-Mage adaptation | Included here |
|---|---|---|---|
| TFN, LMF, LF-DNN, EF-LSTM, MFN, Graph-MFN, MFM, MulT, MTFN, MLF-DNN, MLMF, Self-MM, MMIM | MMSA framework definitions/trainers | fixed-fold adapter and local BERT compatibility | wrapper only; upstream source/license must be obtained separately |
| ConFEDE | official SIMS implementation | paths, dimensions, lengths, positional encoding, Transformers AdamW compatibility; retained contrastive pretraining and TVA fusion | wrapper only |
| CLGSI | official implementation | paths, lengths, feature dimensions | wrapper only |
| KuDA | official implementation | dimensions, local BERT location, Transformer API compatibility | wrapper only |
| CubeMLP | paper-faithful reproduction | CS-Mage fold and feature interface | wrapper included |
| EUAR | paper-faithful reproduction; official anonymous code unavailable | Gaussian experts, top-k routing, KL/load balancing, uncertainty-aware routing | wrapper included |

The upstream URLs/commits and redistribution permissions must be verified before publication. Do not describe all baselines as official implementations.
