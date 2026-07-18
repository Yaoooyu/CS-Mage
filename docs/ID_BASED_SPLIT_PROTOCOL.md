# ID-based split protocol

## Purpose

`splits/id_split_index.csv` is the canonical lookup table for the fixed
sample-ID evaluation protocols used by every MSA and ASR experiment. It records
the assignment of every `sample_id` to each MSA fold and to the ASR split,
allowing users to reproduce the reported partitions without relying on local
paths or data-file order.

## File schema

| Column | Description |
|---|---|
| `sample_id` | Stable CS-Mage sample identifier |
| `multimodal_label` | Published five-level multimodal sentiment class |
| `msa_fold_1` to `msa_fold_5` | The sample's `train`, `validation`, or `test` role in each MSA fold |
| `asr_split` | The sample's `train`, `validation`, or `test` role in the ASR protocol |

Every sample appears in exactly one role for each MSA fold. Across the five
MSA folds, every sample appears in a test role exactly once. The ASR split
contains 760 training, 160 validation, and 165 test samples.

## How to use the index

1. Match each local feature record or audio manifest entry to `sample_id`.
2. For MSA, select the appropriate `msa_fold_k` column and filter rows by
   `train`, `validation`, or `test`.
3. For ASR, filter rows using `asr_split`.
4. Keep the assignments unchanged when reproducing the reported results.

## Protocol scope

The index defines the fixed **sample-ID-level** protocol for MAGE-Fusion,
ablations, all MSA baselines, and all ASR experiments.
