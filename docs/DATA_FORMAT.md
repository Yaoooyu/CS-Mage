# Data format

The dataset download links and data-use terms are documented in
`DATA_ACCESS_AND_ETHICS.md`. The feature package is named
`CS-Mage_fulldata.pkl`. The MSA loader expects the original fold pkl schema
with `text`, `audio`, `vision`, their valid lengths, four label sets,
`raw_text`, and `id`. The observed feature tensors are text 768-D, audio 13-D
MFCC frames, and visual 347-D OpenFace frames; their padded maximum lengths are
39, 553, and 321.

`CS-Mage_fulldata.pkl` contains the complete collection and does not embed the
official train/validation/test assignment. Use the fixed split metadata in
`../splits/msa/` for MSA and `../splits/asr/` for ASR rather than resampling a
new split.

The ASR manifest is a CSV that links each audio record to its reference text
and duration. Build manifests from the downloaded audio data and apply the
fixed split files in `../splits/asr/`.
